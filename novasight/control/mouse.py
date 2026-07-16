from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol

from novasight.control.recoil import FixedRecoilConfig, FixedRecoilController
from novasight.control.registry import CALIBRATED_ANGULAR, UNIVERSAL_SATURATED

CONTROL_MODES = frozenset({CALIBRATED_ANGULAR, UNIVERSAL_SATURATED})
ARRIVAL_CONFIRM_FRAMES = 2
DEPARTURE_CONFIRM_FRAMES = 2


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
    actuation_pending_x: bool = False
    actuation_pending_y: bool = False
    left_trigger_active: bool = False
    left_trigger_hold_ms: float = 0.0
    valid: bool = True
    invalid_reason: str = ""
    reference_x_px: float | None = None
    reference_y_px: float | None = None


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
    recoil_enabled: bool = False
    recoil_start_delay_ms: float = 0.0
    recoil_y_counts_per_observation: float = 0.0


@dataclass(frozen=True, slots=True)
class MouseControllerConfig:
    mode: str
    shared: SharedOutputConfig
    calibrated_angular: CalibratedAngularControllerConfig | None = None
    universal_saturated: UniversalSaturatedControllerConfig | None = None


class ControlController(Protocol):
    mode: str

    def reset(self) -> None: ...

    def compute_counts(self, value: ControllerInput) -> ControllerComputation: ...


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
            if config.calibrated_angular is None:
                raise ValueError("calibrated angular config is required")
            return CalibratedAngularController(config.calibrated_angular)
        if config.mode == UNIVERSAL_SATURATED:
            if config.universal_saturated is None:
                raise ValueError("universal saturated config is required")
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
    settled_x: bool = False
    settled_y: bool = False
    arrival_candidate_x_frames: int = 0
    arrival_candidate_y_frames: int = 0
    departure_candidate_x_frames: int = 0
    departure_candidate_y_frames: int = 0
    previous_observed_error: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    observed_error_history_valid: bool = False

    def reset(self) -> None:
        self.target_id = None
        self.frame_id = None
        self.residual_x_counts = 0.0
        self.residual_y_counts = 0.0
        self.previous_counts = Vec2(0.0, 0.0)
        self.output_history_valid = False
        self.settled_x = False
        self.settled_y = False
        self.arrival_candidate_x_frames = 0
        self.arrival_candidate_y_frames = 0
        self.departure_candidate_x_frames = 0
        self.departure_candidate_y_frames = 0
        self.previous_observed_error = Vec2(0.0, 0.0)
        self.observed_error_history_valid = False


class MouseController:
    """Single production entry that owns exactly one mode controller."""

    def __init__(self, config: MouseControllerConfig) -> None:
        self.config = config
        self.controller = ControllerFactory.create(config)
        self.state = MouseControllerState()
        self.recoil = FixedRecoilController(
            FixedRecoilConfig(
                enabled=config.shared.recoil_enabled,
                start_delay_ms=config.shared.recoil_start_delay_ms,
                y_counts_per_observation=(
                    config.shared.recoil_y_counts_per_observation
                ),
                invert_y=config.shared.invert_y,
            )
        )

    @property
    def mode(self) -> str:
        return self.controller.mode

    def reset(self) -> None:
        self.controller.reset()
        self.state.reset()
        self.recoil.reset()

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

        switched = (
            self.state.target_id is not None and self.state.target_id != observation.target_id
        )
        if switched:
            self.reset()

        center_x = (
            float(observation.reference_x_px)
            if observation.reference_x_px is not None
            else observation.control_width_px * 0.5
        )
        center_y = (
            float(observation.reference_y_px)
            if observation.reference_y_px is not None
            else observation.control_height_px * 0.5
        )
        observed_error = Vec2(
            observation.observed_x_px - center_x,
            observation.observed_y_px - center_y,
        )
        predicted_error = Vec2(
            observation.predicted_x_px - center_x,
            observation.predicted_y_px - center_y,
        )
        crossed_center_x = self.state.observed_error_history_valid and _crossed_center(
            self.state.previous_observed_error.x, observed_error.x
        )
        crossed_center_y = self.state.observed_error_history_valid and _crossed_center(
            self.state.previous_observed_error.y, observed_error.y
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
        (
            self.state.settled_x,
            self.state.arrival_candidate_x_frames,
            self.state.departure_candidate_x_frames,
            arrival_exit_x_px,
        ) = _update_arrival_axis(
            settled=self.state.settled_x,
            candidate_frames=self.state.arrival_candidate_x_frames,
            departure_candidate_frames=self.state.departure_candidate_x_frames,
            observed_error_px=observed_error.x,
            enter_px=shared.deadzone_x_px,
            crossed_center=crossed_center_x,
        )
        (
            self.state.settled_y,
            self.state.arrival_candidate_y_frames,
            self.state.departure_candidate_y_frames,
            arrival_exit_y_px,
        ) = _update_arrival_axis(
            settled=self.state.settled_y,
            candidate_frames=self.state.arrival_candidate_y_frames,
            departure_candidate_frames=self.state.departure_candidate_y_frames,
            observed_error_px=observed_error.y,
            enter_px=shared.deadzone_y_px,
            crossed_center=crossed_center_y,
        )
        hold_x = (
            self.state.settled_x
            or (shared.deadzone_x_px > 0.0 and abs(observed_error.x) <= shared.deadzone_x_px)
            or observation.actuation_pending_x
        )
        hold_y = (
            self.state.settled_y
            or (shared.deadzone_y_px > 0.0 and abs(observed_error.y) <= shared.deadzone_y_px)
            or observation.actuation_pending_y
        )
        deadzone_limited = Vec2(
            0.0 if hold_x else computation.counts.x,
            0.0 if hold_y else computation.counts.y,
        )
        directed_counts = Vec2(
            deadzone_limited.x,
            -deadzone_limited.y if shared.invert_y else deadzone_limited.y,
        )
        recoil = self.recoil.calculate(
            left_trigger_active=(
                observation.left_trigger_active and observation.observed_valid
            ),
            left_trigger_hold_ms=observation.left_trigger_hold_ms,
        )
        mixed_counts = Vec2(
            directed_counts.x,
            directed_counts.y + recoil.emitted_counts_y,
        )
        previous_counts = (
            self.state.previous_counts if self.state.output_history_valid else Vec2(0.0, 0.0)
        )
        slew_limited = Vec2(
            _slew_limit(
                mixed_counts.x,
                previous_counts.x,
                shared.max_count_slew_x,
            ),
            _slew_limit(
                mixed_counts.y,
                previous_counts.y,
                shared.max_count_slew_y,
            ),
        )
        feasible_counts = Vec2(
            _clamp_axis(slew_limited.x, shared.max_budget_counts_x),
            _clamp_axis(slew_limited.y, shared.max_budget_counts_y),
        )

        residual_input_x, residual_direction_reset_x = _residual_for_direction(
            feasible_counts.x,
            self.state.residual_x_counts,
        )
        residual_input_y, residual_direction_reset_y = _residual_for_direction(
            feasible_counts.y,
            self.state.residual_y_counts,
        )
        counts_x, residual_x = _quantize_counts(feasible_counts.x, residual_input_x)
        counts_y, residual_y = _quantize_counts(feasible_counts.y, residual_input_y)
        self.state.residual_x_counts = residual_x
        self.state.residual_y_counts = residual_y
        self.state.target_id = observation.target_id
        self.state.frame_id = observation.frame_id
        self.state.previous_counts = feasible_counts
        self.state.output_history_valid = True
        if observation.observed_valid:
            self.state.previous_observed_error = observed_error
            self.state.observed_error_history_valid = True
        else:
            self.state.observed_error_history_valid = False

        debug = {
            "algorithm": self.mode,
            "control_mode": self.mode,
            "control_reference_x_px": center_x,
            "control_reference_y_px": center_y,
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
            "feedback_demand_y": directed_counts.y,
            "recoil_enabled": shared.recoil_enabled,
            "recoil_active": recoil.active,
            "recoil_left_hold_ms": observation.left_trigger_hold_ms,
            "recoil_y_counts_per_observation": (
                shared.recoil_y_counts_per_observation
            ),
            "recoil_y_counts_float": recoil.requested_counts_y,
            "recoil_y_counts_emitted": recoil.emitted_counts_y,
            "recoil_residual_y_counts": recoil.residual_counts_y,
            "recoil_block_reason": recoil.block_reason,
            "combined_demand_y": directed_counts.y + recoil.requested_counts_y,
            "mixed_counts_x_float": mixed_counts.x,
            "mixed_counts_y_float": mixed_counts.y,
            "slew_limited_counts_x_float": slew_limited.x,
            "slew_limited_counts_y_float": slew_limited.y,
            "slew_policy": "limit_growth_allow_braking_zero_cross",
            "feasible_counts_x_float": feasible_counts.x,
            "feasible_counts_y_float": feasible_counts.y,
            "budget_clamped_x": feasible_counts.x != slew_limited.x,
            "budget_clamped_y": feasible_counts.y != slew_limited.y,
            "residual_x_counts": self.state.residual_x_counts,
            "residual_y_counts": self.state.residual_y_counts,
            "count_quantization": "nearest_integer_with_fractional_residual",
            "arrival_state": (
                "SETTLED"
                if self.state.settled_x and self.state.settled_y
                else "ENTERING"
                if hold_x and hold_y
                else "TRACKING"
            ),
            "arrival_settled_x": self.state.settled_x,
            "arrival_settled_y": self.state.settled_y,
            "arrival_candidate_x_frames": self.state.arrival_candidate_x_frames,
            "arrival_candidate_y_frames": self.state.arrival_candidate_y_frames,
            "arrival_departure_candidate_x_frames": self.state.departure_candidate_x_frames,
            "arrival_departure_candidate_y_frames": self.state.departure_candidate_y_frames,
            "arrival_confirm_frames": ARRIVAL_CONFIRM_FRAMES,
            "departure_confirm_frames": DEPARTURE_CONFIRM_FRAMES,
            "arrival_enter_x_px": shared.deadzone_x_px,
            "arrival_enter_y_px": shared.deadzone_y_px,
            "arrival_exit_x_px": arrival_exit_x_px,
            "arrival_exit_y_px": arrival_exit_y_px,
            "arrival_crossed_center_x": crossed_center_x,
            "arrival_crossed_center_y": crossed_center_y,
            "actuation_pending_x": observation.actuation_pending_x,
            "actuation_pending_y": observation.actuation_pending_y,
            "residual_direction_reset_x": residual_direction_reset_x,
            "residual_direction_reset_y": residual_direction_reset_y,
            "final_dx": counts_x,
            "final_dy": counts_y,
        }
        settled = self.state.settled_x and self.state.settled_y
        accumulating = (
            counts_x == 0
            and counts_y == 0
            and (abs(self.state.residual_x_counts) > 0.0 or abs(self.state.residual_y_counts) > 0.0)
        )
        feedback_pending = (
            counts_x == 0
            and counts_y == 0
            and (observation.actuation_pending_x or observation.actuation_pending_y)
        )
        reason = (
            "AIM_SETTLED"
            if settled
            else "ACTUATION_FEEDBACK_PENDING"
            if feedback_pending
            else "ACCUMULATING_FRACTIONAL_COUNTS"
            if accumulating
            else "mouse_control"
        )
        debug["movement_strategy"] = (
            "hold_position"
            if settled
            else "wait_for_actuation_feedback"
            if feedback_pending
            else "accumulate_fractional_counts"
            if accumulating
            else "predictive_tracking"
        )
        return MoveCommand(
            dx=counts_x,
            dy=counts_y,
            confidence=observation.target_confidence,
            reason=reason,
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


def _quantize_counts(requested: float, residual: float) -> tuple[int, float]:
    if not math.isfinite(requested) or not math.isfinite(residual):
        return 0, 0.0
    if requested == 0.0:
        return 0, 0.0
    total = requested + residual
    quantized = _round_half_away_from_zero(total)
    return quantized, total - quantized


def _residual_for_direction(requested: float, residual: float) -> tuple[float, bool]:
    if not math.isfinite(requested) or not math.isfinite(residual):
        return 0.0, residual != 0.0
    if requested == 0.0 or residual == 0.0 or requested * residual > 0.0:
        return residual, False
    return 0.0, True


def _crossed_center(previous_error_px: float, current_error_px: float) -> bool:
    if not math.isfinite(previous_error_px) or not math.isfinite(current_error_px):
        return False
    if previous_error_px == 0.0 or current_error_px == 0.0:
        return True
    return (previous_error_px < 0.0 < current_error_px) or (
        previous_error_px > 0.0 > current_error_px
    )


def _update_arrival_axis(
    *,
    settled: bool,
    candidate_frames: int,
    departure_candidate_frames: int,
    observed_error_px: float,
    enter_px: float,
    crossed_center: bool,
) -> tuple[bool, int, int, float]:
    enter = max(0.0, float(enter_px))
    exit_threshold = max(enter + 1.0, enter * 1.5) if enter > 0.0 else 0.0
    if enter <= 0.0 or not math.isfinite(observed_error_px):
        return False, 0, 0, exit_threshold
    absolute_error = abs(float(observed_error_px))
    if settled:
        if absolute_error <= exit_threshold:
            return True, ARRIVAL_CONFIRM_FRAMES, 0, exit_threshold
        next_departure_frames = min(
            DEPARTURE_CONFIRM_FRAMES,
            max(0, int(departure_candidate_frames)) + 1,
        )
        if next_departure_frames < DEPARTURE_CONFIRM_FRAMES:
            return True, ARRIVAL_CONFIRM_FRAMES, next_departure_frames, exit_threshold
        return False, 0, 0, exit_threshold
    if crossed_center:
        return True, ARRIVAL_CONFIRM_FRAMES, 0, exit_threshold
    if absolute_error <= enter:
        next_frames = min(ARRIVAL_CONFIRM_FRAMES, max(0, int(candidate_frames)) + 1)
        return next_frames >= ARRIVAL_CONFIRM_FRAMES, next_frames, 0, exit_threshold
    return False, 0, 0, exit_threshold


def _projection_geometry(
    width: float, height: float, fov_x_deg: float
) -> tuple[float, float] | None:
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
    if not math.isfinite(requested) or not math.isfinite(previous):
        return 0.0
    limit = max(0.0, float(max_slew))
    if requested == 0.0:
        return 0.0
    if previous == 0.0:
        return _clamp(requested, -limit, limit)
    if requested * previous < 0.0:
        return 0.0
    if abs(requested) <= abs(previous):
        return requested
    growth = min(abs(requested) - abs(previous), limit)
    return math.copysign(abs(previous) + growth, requested)


def _clamp_axis(value: float | int, limit: float | int) -> float:
    bound = max(0.0, float(limit))
    return _clamp(float(value), -bound, bound)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
