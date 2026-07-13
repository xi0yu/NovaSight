from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import atan, copysign, hypot, isfinite, pi, sqrt, tan, trunc
from typing import Any


ALGORITHM_ID = "dual_phase_atan_predictive_v1"


class ControlMode(Enum):
    FAR = "far"
    NEAR = "near"


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    fov_x_deg: float = 105.0
    counts_per_360: float = 9980.0
    invert_y: bool = False


@dataclass(frozen=True, slots=True)
class ModeSelectorConfig:
    near_enter_min_px: float = 12.0
    near_exit_min_px: float = 18.0
    near_enter_bbox_h_ratio: float = 0.45
    near_exit_bbox_h_ratio: float = 0.60


@dataclass(frozen=True, slots=True)
class AtanPhaseConfig:
    kp: float
    atan_scale_counts: float
    max_counts_per_update: float


def _default_far() -> AtanPhaseConfig:
    return AtanPhaseConfig(kp=0.35, atan_scale_counts=256.0, max_counts_per_update=140.0)


def _default_near() -> AtanPhaseConfig:
    return AtanPhaseConfig(kp=0.15, atan_scale_counts=256.0, max_counts_per_update=60.0)


@dataclass(frozen=True, slots=True)
class EstimatorConfig:
    measurement_std_px: float = 1.5
    acceleration_std_px_s2: float = 600.0
    min_dt_s: float = 0.003
    reset_dt_s: float = 0.080
    innovation_soft_gate_sigma: float = 3.0
    innovation_hard_gate_sigma: float = 8.0
    hard_outlier_reset_count: int = 2
    max_velocity_px_s: float = 3000.0
    warmup_updates: int = 4


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    enabled_x: bool = True
    enabled_y: bool = False
    actuation_delay_s: float = 0.005
    max_horizon_s: float = 0.035
    far_weight: float = 0.30
    near_weight: float = 0.12
    far_abs_cap_px: float = 8.0
    near_abs_cap_px: float = 2.0
    far_base_cap_px: float = 1.0
    near_base_cap_px: float = 0.5
    far_relative_cap: float = 0.25
    near_relative_cap: float = 0.15
    near_cross_allow_px: float = 0.5
    high_confidence_cross_threshold: float = 0.85
    overzero_cooldown_frames: int = 2


@dataclass(frozen=True, slots=True)
class QuantizerConfig:
    min_effective_counts: int = 1


@dataclass(frozen=True, slots=True)
class DualPhaseAtanPredictiveV1Config:
    freshness_threshold_ms: float = 55.0
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    mode: ModeSelectorConfig = field(default_factory=ModeSelectorConfig)
    far: AtanPhaseConfig = field(default_factory=_default_far)
    near: AtanPhaseConfig = field(default_factory=_default_near)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)


@dataclass(frozen=True, slots=True)
class DualPhaseAtanPredictiveV1Observation:
    generation: int
    frame_id: int
    target_id: int
    capture_ts_ns: int
    inference_end_ts_ns: int
    control_now_ns: int
    aim_x: float
    aim_y: float
    crosshair_x: float
    crosshair_y: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    observation_width: int
    observation_height: int
    roi_left: int
    roi_top: int
    roi_width: int
    roi_height: int
    source_width: int
    source_height: int
    detection_confidence: float
    track_confidence: float
    trigger_active: bool
    target_valid: bool


@dataclass(frozen=True, slots=True)
class ControlDecision:
    dx: int
    dy: int
    emit_allowed: bool
    block_reason: str
    telemetry: dict[str, Any]


class DualPhaseAtanPredictiveV1Algorithm:
    algorithm_id = ALGORITHM_ID

    def __init__(self, config: DualPhaseAtanPredictiveV1Config) -> None:
        self.config = config
        self._mode = ControlMode.FAR
        self._target_id: int | None = None
        self._last_generation: int | None = None
        self._last_frame_id: int | None = None
        self._last_capture_ts_ns: int | None = None
        self._quantizer_x = _AxisQuantizer(config.quantizer.min_effective_counts)
        self._quantizer_y = _AxisQuantizer(config.quantizer.min_effective_counts)
        self._estimator_x = _ConstantVelocityKalman1D(config.estimator)
        self._prediction_cooldown_frames = 0
        self._previous_error_real_x = 0.0
        self._previous_error_real_y = 0.0
        self._real_error_history_valid = False
        self._previous_estimated_velocity_x = 0.0
        self._direction_stable_updates = 0

    def reset(self) -> None:
        self._reset_target_state()
        self._last_generation = None
        self._last_frame_id = None
        self._last_capture_ts_ns = None

    def _reset_target_state(self) -> None:
        self._mode = ControlMode.FAR
        self._target_id = None
        self._quantizer_x.reset()
        self._quantizer_y.reset()
        self._estimator_x.clear()
        self._prediction_cooldown_frames = 0
        self._previous_error_real_x = 0.0
        self._previous_error_real_y = 0.0
        self._real_error_history_valid = False
        self._previous_estimated_velocity_x = 0.0
        self._direction_stable_updates = 0

    def release_trigger(self) -> None:
        """Discard fractional output without resetting motion observations."""
        self._quantizer_x.reset()
        self._quantizer_y.reset()

    def calculate(
        self,
        observation: DualPhaseAtanPredictiveV1Observation,
    ) -> ControlDecision:
        frame_age_ns = int(observation.control_now_ns) - int(observation.capture_ts_ns)
        inference_end_ns = int(observation.inference_end_ts_ns)
        if (
            frame_age_ns < 0
            or inference_end_ns < int(observation.capture_ts_ns)
            or inference_end_ns > int(observation.control_now_ns)
        ):
            return self._blocked_decision(
                observation,
                "TIMESTAMP_DOMAIN_INVALID",
                frame_age_ms=frame_age_ns / 1_000_000.0,
            )
        frame_age_ms = frame_age_ns / 1_000_000.0
        if frame_age_ms > self.config.freshness_threshold_ms:
            return self._blocked_decision(
                observation,
                "STALE_OBSERVATION",
                frame_age_ms=frame_age_ms,
            )
        if (
            (self._last_generation is not None and observation.generation <= self._last_generation)
            or (self._last_frame_id is not None and observation.frame_id <= self._last_frame_id)
            or (
                self._last_capture_ts_ns is not None
                and observation.capture_ts_ns <= self._last_capture_ts_ns
            )
        ):
            return self._blocked_decision(
                observation,
                "NON_MONOTONIC_OBSERVATION",
                frame_age_ms=frame_age_ms,
            )

        self._last_generation = observation.generation
        self._last_frame_id = observation.frame_id
        self._last_capture_ts_ns = observation.capture_ts_ns
        if self._target_id is not None and observation.target_id != self._target_id:
            self._reset_target_state()
        if not observation.target_valid:
            self._reset_target_state()
            return self._blocked_decision(observation, "TARGET_INVALID")
        if not _valid_observation_geometry(observation):
            return self._blocked_decision(observation, "GEOMETRY_INVALID")

        self._target_id = observation.target_id

        error_x = float(observation.aim_x - observation.crosshair_x)
        error_y = float(observation.aim_y - observation.crosshair_y)
        crossed_x = self._real_error_history_valid and _crossed_center(
            self._previous_error_real_x,
            error_x,
        )
        crossed_y = self._real_error_history_valid and _crossed_center(
            self._previous_error_real_y,
            error_y,
        )
        cleared_residual_x = self._quantizer_x.accumulator if crossed_x else 0.0
        cleared_residual_y = self._quantizer_y.accumulator if crossed_y else 0.0
        if crossed_x:
            self._quantizer_x.reset()
            self._estimator_x.damp_velocity(0.25)
            self._prediction_cooldown_frames = max(
                self._prediction_cooldown_frames,
                max(0, self.config.prediction.overzero_cooldown_frames),
            )
        if crossed_y:
            self._quantizer_y.reset()
        bbox_height = max(0.0, float(observation.bbox_y2 - observation.bbox_y1))
        distance = hypot(error_x, error_y)
        enter = max(
            self.config.mode.near_enter_min_px,
            bbox_height * self.config.mode.near_enter_bbox_h_ratio,
        )
        exit_threshold = max(
            self.config.mode.near_exit_min_px,
            bbox_height * self.config.mode.near_exit_bbox_h_ratio,
        )
        if self._mode is ControlMode.FAR and distance <= enter:
            self._mode = ControlMode.NEAR
        elif self._mode is ControlMode.NEAR and distance >= exit_threshold:
            self._mode = ControlMode.FAR

        motion = self._estimator_x.update(
            measurement=observation.aim_x,
            timestamp_ns=observation.capture_ts_ns,
            detection_confidence=observation.detection_confidence,
        )
        velocity_sign = _sign(motion.velocity, epsilon=1.0)
        previous_velocity_sign = _sign(
            self._previous_estimated_velocity_x,
            epsilon=1.0,
        )
        if not motion.accepted or motion.reset or velocity_sign == 0:
            self._direction_stable_updates = 0
        elif previous_velocity_sign in {0, velocity_sign}:
            self._direction_stable_updates += 1
        else:
            self._direction_stable_updates = 0
        self._previous_estimated_velocity_x = motion.velocity
        direction_quality = _clamp(
            self._direction_stable_updates / max(1, self.config.estimator.warmup_updates - 1),
            0.0,
            1.0,
        )
        age_s = max(
            0.0,
            (observation.control_now_ns - observation.capture_ts_ns) / 1_000_000_000.0,
        )
        horizon_s = _clamp(
            age_s + self.config.prediction.actuation_delay_s,
            0.0,
            max(0.0, self.config.prediction.max_horizon_s),
        )
        prediction_mode_weight = (
            self.config.prediction.far_weight
            if self._mode is ControlMode.FAR
            else self.config.prediction.near_weight
        )
        track_confidence = _clamp(observation.track_confidence, 0.0, 1.0)
        prediction_confidence = motion.confidence * track_confidence * direction_quality
        prediction_cooldown_active = self._prediction_cooldown_frames > 0
        prediction_allowed = self.config.prediction.enabled_x and not prediction_cooldown_active
        prediction_weight = (
            prediction_mode_weight * prediction_confidence if prediction_allowed else 0.0
        )
        prediction_raw_offset_x = motion.velocity * horizon_s
        prediction_weighted_offset_x = prediction_raw_offset_x * prediction_weight
        if self._mode is ControlMode.FAR:
            absolute_cap = self.config.prediction.far_abs_cap_px
            base_cap = self.config.prediction.far_base_cap_px
            relative_cap = self.config.prediction.far_relative_cap
        else:
            absolute_cap = self.config.prediction.near_abs_cap_px
            base_cap = self.config.prediction.near_base_cap_px
            relative_cap = self.config.prediction.near_relative_cap
        prediction_allowed_cap_x = min(
            max(0.0, absolute_cap),
            max(0.0, base_cap + relative_cap * abs(error_x)),
        )
        prediction_safe_offset_x = _clamp(
            prediction_weighted_offset_x,
            -prediction_allowed_cap_x,
            prediction_allowed_cap_x,
        )
        control_error_x = error_x + prediction_safe_offset_x
        prediction_crossing_limited = False
        if (
            self._mode is ControlMode.NEAR
            and _sign(error_x) != 0
            and _sign(control_error_x) != 0
            and _sign(error_x) != _sign(control_error_x)
        ):
            prediction_crossing_limited = True
            if prediction_confidence < self.config.prediction.high_confidence_cross_threshold:
                control_error_x = 0.0
            else:
                control_error_x = copysign(
                    min(
                        abs(control_error_x),
                        max(0.0, self.config.prediction.near_cross_allow_px),
                    ),
                    control_error_x,
                )
            prediction_safe_offset_x = control_error_x - error_x
        control_error_y = error_y
        source_error_x = control_error_x * observation.roi_width / observation.observation_width
        source_error_y = control_error_y * observation.roi_height / observation.observation_height
        fov_x_rad = self.config.projection.fov_x_deg * pi / 180.0
        focal_x = (observation.source_width * 0.5) / tan(fov_x_rad * 0.5)
        focal_y = focal_x
        counts_per_rad = self.config.projection.counts_per_360 / (2.0 * pi)
        full_counts_x = atan(source_error_x / focal_x) * counts_per_rad
        full_counts_y = atan(source_error_y / focal_y) * counts_per_rad
        if self.config.projection.invert_y:
            full_counts_y = -full_counts_y

        phase = self.config.far if self._mode is ControlMode.FAR else self.config.near
        kp = _clamp(phase.kp, 0.0, 1.0)
        demand_x = _atan_demand(full_counts_x, kp, phase)
        demand_y = _atan_demand(full_counts_y, kp, phase)
        if observation.trigger_active:
            dx, residual_direction_reset_x = self._quantizer_x.quantize(demand_x)
            dy, residual_direction_reset_y = self._quantizer_y.quantize(demand_y)
            block_reason = ""
        else:
            self._quantizer_x.reset()
            self._quantizer_y.reset()
            dx = 0
            dy = 0
            residual_direction_reset_x = False
            residual_direction_reset_y = False
            block_reason = "TRIGGER_INACTIVE"
        emit_allowed = not block_reason and (dx != 0 or dy != 0)
        self._previous_error_real_x = error_x
        self._previous_error_real_y = error_y
        self._real_error_history_valid = True
        if self._prediction_cooldown_frames > 0:
            self._prediction_cooldown_frames -= 1

        return ControlDecision(
            dx=dx,
            dy=dy,
            emit_allowed=emit_allowed,
            block_reason=block_reason,
            telemetry={
                "algorithm_id": ALGORITHM_ID,
                "generation": observation.generation,
                "frame_id": observation.frame_id,
                "target_id": observation.target_id,
                "capture_ts_ns": observation.capture_ts_ns,
                "inference_end_ts_ns": observation.inference_end_ts_ns,
                "control_now_ns": observation.control_now_ns,
                "frame_age_ms": frame_age_ms,
                "measurement_dt_ms": motion.dt_s * 1000.0,
                "aim_x": observation.aim_x,
                "aim_y": observation.aim_y,
                "bbox_x1": observation.bbox_x1,
                "bbox_y1": observation.bbox_y1,
                "bbox_x2": observation.bbox_x2,
                "bbox_y2": observation.bbox_y2,
                "bbox_width": max(0.0, observation.bbox_x2 - observation.bbox_x1),
                "bbox_height": bbox_height,
                "detection_confidence": observation.detection_confidence,
                "track_confidence": track_confidence,
                "error_real_x": error_x,
                "error_real_y": error_y,
                "mode": self._mode.value,
                "near_enter_threshold_px": enter,
                "near_exit_threshold_px": exit_threshold,
                "measurement_dt_s": motion.dt_s,
                "kp": kp,
                "estimated_position_x": motion.position,
                "estimated_velocity_x": motion.velocity,
                "innovation_x": motion.innovation,
                "normalized_innovation_x": motion.normalized_innovation,
                "motion_confidence": motion.confidence,
                "direction_quality": direction_quality,
                "direction_stable_updates": self._direction_stable_updates,
                "prediction_confidence": prediction_confidence,
                "prediction_mode_weight": prediction_mode_weight,
                "estimator_accepted": motion.accepted,
                "estimator_reset": motion.reset,
                "prediction_horizon_s": horizon_s,
                "prediction_horizon_ms": horizon_s * 1000.0,
                "prediction_raw_offset_x": prediction_raw_offset_x,
                "prediction_weight": prediction_weight,
                "prediction_allowed_cap_x": prediction_allowed_cap_x,
                "prediction_safe_offset_x": prediction_safe_offset_x,
                "prediction_crossing_limited": prediction_crossing_limited,
                "prediction_allowed": prediction_allowed,
                "prediction_cooldown_active": prediction_cooldown_active,
                "prediction_cooldown_frames_remaining": self._prediction_cooldown_frames,
                "overzero_detected_x": crossed_x,
                "overzero_detected_y": crossed_y,
                "overzero_cleared_residual_x": cleared_residual_x,
                "overzero_cleared_residual_y": cleared_residual_y,
                "error_control_x": control_error_x,
                "error_control_y": control_error_y,
                "full_error_counts_x": full_counts_x,
                "full_error_counts_y": full_counts_y,
                "float_demand_x": demand_x,
                "float_demand_y": demand_y,
                "integer_command_x": dx,
                "integer_command_y": dy,
                "quantizer_residual_x": self._quantizer_x.accumulator,
                "quantizer_residual_y": self._quantizer_y.accumulator,
                "residual_direction_reset_x": residual_direction_reset_x,
                "residual_direction_reset_y": residual_direction_reset_y,
                "will_emit": emit_allowed,
                "block_reason": block_reason,
            },
        )

    def _blocked_decision(
        self,
        observation: DualPhaseAtanPredictiveV1Observation,
        reason: str,
        *,
        frame_age_ms: float | None = None,
    ) -> ControlDecision:
        self._quantizer_x.reset()
        self._quantizer_y.reset()
        return ControlDecision(
            dx=0,
            dy=0,
            emit_allowed=False,
            block_reason=reason,
            telemetry={
                "algorithm_id": ALGORITHM_ID,
                "generation": observation.generation,
                "frame_id": observation.frame_id,
                "target_id": observation.target_id,
                "capture_ts_ns": observation.capture_ts_ns,
                "inference_end_ts_ns": observation.inference_end_ts_ns,
                "control_now_ns": observation.control_now_ns,
                "frame_age_ms": frame_age_ms,
                "aim_x": observation.aim_x,
                "aim_y": observation.aim_y,
                "bbox_x1": observation.bbox_x1,
                "bbox_y1": observation.bbox_y1,
                "bbox_x2": observation.bbox_x2,
                "bbox_y2": observation.bbox_y2,
                "detection_confidence": observation.detection_confidence,
                "track_confidence": observation.track_confidence,
                "mode": self._mode.value,
                "will_emit": False,
                "block_reason": reason,
                "quantizer_residual_x": 0.0,
                "quantizer_residual_y": 0.0,
            },
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _atan_demand(
    full_error_counts: float,
    kp: float,
    config: AtanPhaseConfig,
) -> float:
    scale = float(config.atan_scale_counts)
    if not isfinite(scale) or scale <= 0.0:
        return 0.0
    value = kp * scale * atan(full_error_counts / scale)
    maximum = max(0.0, float(config.max_counts_per_update))
    return _clamp(value, -maximum, maximum)


class _AxisQuantizer:
    def __init__(self, min_effective_counts: int) -> None:
        self.min_effective_counts = max(1, int(min_effective_counts))
        self.accumulator = 0.0

    def reset(self) -> None:
        self.accumulator = 0.0

    def quantize(self, demand: float) -> tuple[int, bool]:
        if not isfinite(demand):
            self.reset()
            return 0, True
        direction_reset = (
            self.accumulator != 0.0 and demand != 0.0 and self.accumulator * demand < 0.0
        )
        if direction_reset:
            self.accumulator = 0.0
        self.accumulator += demand
        integer_value = trunc(self.accumulator)
        if abs(integer_value) < self.min_effective_counts:
            return 0, direction_reset
        self.accumulator -= integer_value
        return int(integer_value), direction_reset


@dataclass(frozen=True, slots=True)
class _MotionEstimate:
    position: float
    velocity: float
    confidence: float
    innovation: float
    normalized_innovation: float
    dt_s: float
    accepted: bool
    reset: bool


class _ConstantVelocityKalman1D:
    def __init__(self, config: EstimatorConfig) -> None:
        self.config = config
        self.clear()

    def clear(self) -> None:
        self.initialized = False
        self.position = 0.0
        self.velocity = 0.0
        self.p00 = 25.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 10_000.0
        self.last_ts_ns = 0
        self.update_count = 0
        self.outlier_streak = 0

    def reset(self, position: float, timestamp_ns: int) -> None:
        self.initialized = True
        self.position = float(position)
        self.velocity = 0.0
        self.p00 = 25.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 10_000.0
        self.last_ts_ns = int(timestamp_ns)
        self.update_count = 1
        self.outlier_streak = 0

    def damp_velocity(self, factor: float) -> None:
        self.velocity *= _clamp(float(factor), 0.0, 1.0)

    def update(
        self,
        *,
        measurement: float,
        timestamp_ns: int,
        detection_confidence: float,
    ) -> _MotionEstimate:
        confidence = _clamp(float(detection_confidence), 0.05, 1.0)
        if not self.initialized:
            self.reset(measurement, timestamp_ns)
            return self._result(0.0, 0.0, 0.0, True, True)

        dt_s = (int(timestamp_ns) - self.last_ts_ns) / 1_000_000_000.0
        if dt_s <= 0.0:
            return self._result(0.0, 0.0, dt_s, False, False)
        if dt_s < self.config.min_dt_s:
            return self._result(
                float(measurement) - self.position,
                0.0,
                dt_s,
                False,
                False,
            )
        if dt_s > self.config.reset_dt_s:
            self.reset(measurement, timestamp_ns)
            return self._result(0.0, 0.0, dt_s, True, True)

        predicted_position = self.position + self.velocity * dt_s
        predicted_velocity = self.velocity
        accel_var = self.config.acceleration_std_px_s2**2
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt2 * dt2
        q00 = 0.25 * dt4 * accel_var
        q01 = 0.5 * dt3 * accel_var
        q11 = dt2 * accel_var
        predicted_p00 = self.p00 + dt_s * (self.p01 + self.p10) + dt2 * self.p11 + q00
        predicted_p01 = self.p01 + dt_s * self.p11 + q01
        predicted_p10 = self.p10 + dt_s * self.p11 + q01
        predicted_p11 = self.p11 + q11

        measurement_var = self.config.measurement_std_px**2 / confidence**2
        innovation = float(measurement) - predicted_position
        innovation_variance = max(predicted_p00 + measurement_var, 1e-9)
        innovation_sigma = sqrt(innovation_variance)
        normalized_innovation = innovation / innovation_sigma
        if abs(normalized_innovation) > self.config.innovation_hard_gate_sigma:
            self.outlier_streak += 1
            if self.outlier_streak >= self.config.hard_outlier_reset_count:
                self.reset(measurement, timestamp_ns)
                return self._result(
                    innovation,
                    normalized_innovation,
                    dt_s,
                    False,
                    True,
                )
            self.position = predicted_position
            self.velocity = predicted_velocity
            self.p00 = predicted_p00
            self.p01 = predicted_p01
            self.p10 = predicted_p10
            self.p11 = predicted_p11
            self.last_ts_ns = int(timestamp_ns)
            return self._result(
                innovation,
                normalized_innovation,
                dt_s,
                False,
                False,
            )

        self.outlier_streak = 0
        innovation_limit = self.config.innovation_soft_gate_sigma * innovation_sigma
        safe_innovation = _clamp(innovation, -innovation_limit, innovation_limit)
        gain_position = predicted_p00 / innovation_variance
        gain_velocity = predicted_p10 / innovation_variance
        self.position = predicted_position + gain_position * safe_innovation
        self.velocity = _clamp(
            predicted_velocity + gain_velocity * safe_innovation,
            -self.config.max_velocity_px_s,
            self.config.max_velocity_px_s,
        )
        # Joseph-form covariance update for H=[1, 0]. This preserves symmetry
        # and positive semidefiniteness better than the shortened P=(I-KH)P
        # update during long high-frequency runs.
        one_minus_kp = 1.0 - gain_position
        new_p00 = (
            one_minus_kp * one_minus_kp * predicted_p00
            + gain_position * gain_position * measurement_var
        )
        new_p01 = (
            one_minus_kp * (predicted_p01 - gain_velocity * predicted_p00)
            + gain_position * gain_velocity * measurement_var
        )
        new_p10 = (
            one_minus_kp * (predicted_p10 - gain_velocity * predicted_p00)
            + gain_position * gain_velocity * measurement_var
        )
        new_p11 = (
            predicted_p11
            - gain_velocity * predicted_p01
            - gain_velocity * predicted_p10
            + gain_velocity * gain_velocity * predicted_p00
            + gain_velocity * gain_velocity * measurement_var
        )
        symmetric_p01 = 0.5 * (new_p01 + new_p10)
        self.p00 = max(new_p00, 1e-9)
        self.p01 = symmetric_p01
        self.p10 = symmetric_p01
        self.p11 = max(new_p11, 1e-9)
        self.last_ts_ns = int(timestamp_ns)
        self.update_count += 1

        warmup_quality = _clamp(
            self.update_count / max(1, self.config.warmup_updates),
            0.0,
            1.0,
        )
        hard_gate = max(self.config.innovation_hard_gate_sigma, 1e-9)
        innovation_quality = _clamp(
            1.0 - abs(normalized_innovation) / hard_gate,
            0.0,
            1.0,
        )
        motion_confidence = confidence * warmup_quality * innovation_quality
        return self._result(
            innovation,
            normalized_innovation,
            dt_s,
            True,
            False,
            confidence=motion_confidence,
        )

    def _result(
        self,
        innovation: float,
        normalized_innovation: float,
        dt_s: float,
        accepted: bool,
        reset: bool,
        *,
        confidence: float = 0.0,
    ) -> _MotionEstimate:
        return _MotionEstimate(
            position=self.position,
            velocity=self.velocity,
            confidence=confidence,
            innovation=innovation,
            normalized_innovation=normalized_innovation,
            dt_s=dt_s,
            accepted=accepted,
            reset=reset,
        )


def _sign(value: float, epsilon: float = 1e-9) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _crossed_center(
    previous_error: float,
    current_error: float,
) -> bool:
    return (
        isfinite(previous_error)
        and isfinite(current_error)
        and previous_error * current_error < 0.0
    )


def _valid_observation_geometry(
    observation: DualPhaseAtanPredictiveV1Observation,
) -> bool:
    dimensions = (
        observation.observation_width,
        observation.observation_height,
        observation.roi_width,
        observation.roi_height,
        observation.source_width,
        observation.source_height,
    )
    coordinates = (
        observation.aim_x,
        observation.aim_y,
        observation.crosshair_x,
        observation.crosshair_y,
        observation.bbox_x1,
        observation.bbox_y1,
        observation.bbox_x2,
        observation.bbox_y2,
        observation.detection_confidence,
        observation.track_confidence,
    )
    return (
        all(int(value) > 0 for value in dimensions)
        and all(isfinite(float(value)) for value in coordinates)
        and observation.bbox_x2 > observation.bbox_x1
        and observation.bbox_y2 > observation.bbox_y1
    )
