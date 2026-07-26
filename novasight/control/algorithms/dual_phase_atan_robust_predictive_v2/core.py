from __future__ import annotations

from math import atan, hypot, isfinite, pi, tan, trunc

from novasight.control.output import MAX_ABS_MOUSE_MOVE_COUNT
from .models import (
    ALGORITHM_ID,
    AtanModeConfig,
    ControlDecision,
    ControlMode,
    DualPhaseAtanRobustPredictiveV2Config,
    DualPhaseAtanRobustPredictiveV2Observation,
    PredictionResult,
)
from .motion_history import RobustVelocityEstimator


class DualPhaseAtanRobustPredictiveV2Algorithm:
    algorithm_id = ALGORITHM_ID

    def __init__(self, config: DualPhaseAtanRobustPredictiveV2Config) -> None:
        _validate_config(config)
        self.config = config
        self._target_id: int | None = None
        self._last_generation: int | None = None
        self._last_frame_id: int | None = None
        self._last_capture_ts_ns: int | None = None
        self._geometry_signature: tuple[int, ...] | None = None
        self._quantizer_x = _AxisQuantizer()
        self._quantizer_y = _AxisQuantizer()
        self._velocity_x = RobustVelocityEstimator(config.velocity)
        self._previous_error_meas_x = 0.0
        self._previous_error_meas_y = 0.0
        self._measured_error_history_valid = False

    def reset(self) -> None:
        self._reset_target_state()
        self._last_generation = None
        self._last_frame_id = None
        self._last_capture_ts_ns = None
        self._geometry_signature = None

    def _reset_target_state(self) -> None:
        self._target_id = None
        self._quantizer_x.reset()
        self._quantizer_y.reset()
        self._velocity_x.reset()
        self._previous_error_meas_x = 0.0
        self._previous_error_meas_y = 0.0
        self._measured_error_history_valid = False

    def release_trigger(self) -> None:
        """Clear unsent fractional counts without discarding motion history."""

        self._quantizer_x.reset()
        self._quantizer_y.reset()

    def calculate(
        self,
        observation: DualPhaseAtanRobustPredictiveV2Observation,
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
            self._last_generation is not None and observation.generation <= self._last_generation
        ) or (self._last_frame_id is not None and observation.frame_id <= self._last_frame_id):
            return self._blocked_decision(
                observation,
                "NON_MONOTONIC_OBSERVATION",
                frame_age_ms=frame_age_ms,
            )
        capture_timestamp_discontinuity = bool(
            self._last_capture_ts_ns is not None
            and observation.capture_ts_ns <= self._last_capture_ts_ns
        )
        if capture_timestamp_discontinuity:
            self._velocity_x.reset(self._target_id)
            self._measured_error_history_valid = False

        self._last_generation = observation.generation
        self._last_frame_id = observation.frame_id
        self._last_capture_ts_ns = observation.capture_ts_ns

        if (
            self._target_id is not None and observation.target_id != self._target_id
        ) or observation.track_rebuilt:
            self._reset_target_state()
        if not observation.target_valid:
            self._reset_target_state()
            return self._blocked_decision(observation, "TARGET_INVALID", frame_age_ms=frame_age_ms)
        if not _valid_observation_geometry(observation):
            self._reset_target_state()
            return self._blocked_decision(
                observation,
                "GEOMETRY_INVALID",
                frame_age_ms=frame_age_ms,
            )

        geometry_signature = _observation_geometry_signature(observation)
        if self._geometry_signature is not None and geometry_signature != self._geometry_signature:
            self._reset_target_state()
        self._geometry_signature = geometry_signature
        self._target_id = observation.target_id

        error_meas_x = float(observation.aim_x - observation.crosshair_x)
        error_meas_y = float(observation.aim_y - observation.crosshair_y)
        crossed_x = self._measured_error_history_valid and _crossed_center(
            self._previous_error_meas_x,
            error_meas_x,
        )
        crossed_y = self._measured_error_history_valid and _crossed_center(
            self._previous_error_meas_y,
            error_meas_y,
        )
        cleared_residual_x = self._quantizer_x.accumulator if crossed_x else 0.0
        cleared_residual_y = self._quantizer_y.accumulator if crossed_y else 0.0
        if crossed_x:
            self._quantizer_x.reset()
        if crossed_y:
            self._quantizer_y.reset()

        distance = hypot(error_meas_x, error_meas_y)
        mode = (
            ControlMode.NEAR if distance <= self.config.mode.near_threshold_px else ControlMode.FAR
        )

        estimate = None
        if not capture_timestamp_discontinuity and self.config.prediction.enabled:
            estimate = self._velocity_x.update(
                target_id=observation.target_id,
                aim_x=observation.aim_x,
                capture_ts_ns=observation.capture_ts_ns,
                detection_confidence=observation.detection_confidence,
                track_confidence=observation.track_confidence,
            )
        filtered_velocity_x = estimate.filtered_velocity if estimate is not None else 0.0
        motion_confidence = estimate.motion_confidence if estimate is not None else 0.0
        prediction = self._calculate_prediction(
            mode=mode,
            error_meas_x=error_meas_x,
            filtered_velocity_x=filtered_velocity_x,
            motion_confidence=motion_confidence,
            reference_dt_ms=(estimate.reference_dt_ms if estimate is not None else 0.0),
            estimate_available=estimate is not None,
        )
        error_ctrl_x = error_meas_x + prediction.safe_offset_x
        error_ctrl_y = error_meas_y

        source_error_x = error_ctrl_x * observation.roi_width / observation.observation_width
        source_error_y = error_ctrl_y * observation.roi_height / observation.observation_height
        fov_x_rad = self.config.projection.fov_x_deg * pi / 180.0
        focal_x_px = (observation.source_width * 0.5) / tan(fov_x_rad * 0.5)
        # Square source pixels imply the same focal length in X/Y pixel units.
        focal_y_px = focal_x_px
        counts_per_rad = self.config.projection.counts_per_360 / (2.0 * pi)
        full_counts_x = atan(source_error_x / focal_x_px) * counts_per_rad
        full_counts_y = atan(source_error_y / focal_y_px) * counts_per_rad
        if self.config.projection.invert_y:
            full_counts_y = -full_counts_y

        atan_mode = self.config.atan.far if mode is ControlMode.FAR else self.config.atan.near
        demand_x = _atan_demand(full_counts_x, atan_mode, self.config.atan.scale_counts)
        feedback_demand_y = _atan_demand(full_counts_y, atan_mode, self.config.atan.scale_counts)
        humanized_debug = {
            "humanized_motion_enabled": False,
            "humanized_motion_reason": "atan_only_production_path",
        }
        demand_y = _clamp(
            feedback_demand_y,
            -atan_mode.max_counts_per_update,
            atan_mode.max_counts_per_update,
        )
        if observation.trigger_active:
            dx, residual_direction_reset_x = self._quantizer_x.quantize(demand_x)
            dy, residual_direction_reset_y = self._quantizer_y.quantize(demand_y)
            block_reason = ""
        else:
            self.release_trigger()
            dx = 0
            dy = 0
            residual_direction_reset_x = False
            residual_direction_reset_y = False
            block_reason = "TRIGGER_INACTIVE"

        emit_allowed = not block_reason and (dx != 0 or dy != 0)
        self._previous_error_meas_x = error_meas_x
        self._previous_error_meas_y = error_meas_y
        self._measured_error_history_valid = True
        raw_velocities = estimate.raw_velocities if estimate is not None else (None, None, None)

        return ControlDecision(
            dx=dx,
            dy=dy,
            emit_allowed=emit_allowed,
            block_reason=block_reason,
            telemetry={
                "algorithm_id": ALGORITHM_ID,
                **humanized_debug,
                "generation": observation.generation,
                "frame_id": observation.frame_id,
                "target_id": observation.target_id,
                "capture_ts_ns": observation.capture_ts_ns,
                "inference_end_ts_ns": observation.inference_end_ts_ns,
                "control_now_ns": observation.control_now_ns,
                "frame_age_ms": frame_age_ms,
                "measurement_dt_ms": (estimate.measurement_dt_ms if estimate is not None else None),
                "reference_dt_ms": prediction.reference_dt_ms,
                "aim_x": observation.aim_x,
                "aim_y": observation.aim_y,
                "bbox_x1": observation.bbox_x1,
                "bbox_y1": observation.bbox_y1,
                "bbox_x2": observation.bbox_x2,
                "bbox_y2": observation.bbox_y2,
                "bbox_width": observation.bbox_x2 - observation.bbox_x1,
                "bbox_height": observation.bbox_y2 - observation.bbox_y1,
                "detection_confidence": observation.detection_confidence,
                "track_confidence": _clamp(observation.track_confidence, 0.0, 1.0),
                "track_rebuilt": observation.track_rebuilt,
                "capture_timestamp_discontinuity": capture_timestamp_discontinuity,
                "error_meas_x": error_meas_x,
                "error_meas_y": error_meas_y,
                # Compatibility aliases for the shared trace/runtime envelope.
                "error_real_x": error_meas_x,
                "error_real_y": error_meas_y,
                "error_ctrl_x": error_ctrl_x,
                "error_ctrl_y": error_ctrl_y,
                "error_control_x": error_ctrl_x,
                "error_control_y": error_ctrl_y,
                "mode": mode.value,
                "near_threshold_px": self.config.mode.near_threshold_px,
                "history_position_count": self._velocity_x.history_position_count,
                "velocity_1": raw_velocities[0],
                "velocity_2": raw_velocities[1],
                "velocity_3": raw_velocities[2],
                "median_velocity": estimate.median_velocity if estimate is not None else None,
                "filtered_velocity": filtered_velocity_x,
                "estimated_velocity_x": filtered_velocity_x,
                "velocity_unit": "px/ms",
                "velocity_spread": estimate.spread if estimate is not None else None,
                "history_quality": estimate.history_quality if estimate is not None else 0.0,
                "spread_quality": estimate.spread_quality if estimate is not None else 0.0,
                "trend_quality": estimate.trend_quality if estimate is not None else 0.0,
                "detection_quality": (estimate.detection_quality if estimate is not None else 0.0),
                "track_quality": estimate.track_quality if estimate is not None else 0.0,
                "motion_confidence": motion_confidence,
                "prediction_confidence": motion_confidence,
                "prediction_lead_frames": prediction.lead_frames,
                "prediction_raw_offset_x": prediction.raw_offset_x,
                "prediction_weighted_offset_x": prediction.weighted_offset_x,
                "prediction_weight": motion_confidence,
                "prediction_allowed_cap_x": prediction.allowed_cap_x,
                "prediction_safe_offset_x": prediction.safe_offset_x,
                "prediction_allowed": prediction.allowed,
                "overzero_detected_x": crossed_x,
                "overzero_detected_y": crossed_y,
                "overzero_cleared_residual_x": cleared_residual_x,
                "overzero_cleared_residual_y": cleared_residual_y,
                "full_error_counts_x": full_counts_x,
                "full_error_counts_y": full_counts_y,
                "float_demand_x": demand_x,
                "float_demand_y": demand_y,
                "feedback_demand_y": feedback_demand_y,
                "combined_demand_y": demand_y,
                "integer_command_x": dx,
                "integer_command_y": dy,
                "quantizer_residual_x": self._quantizer_x.accumulator,
                "quantizer_residual_y": self._quantizer_y.accumulator,
                "residual_direction_reset_x": residual_direction_reset_x,
                "residual_direction_reset_y": residual_direction_reset_y,
                "trigger_active": observation.trigger_active,
                "executor_success": None,
                "will_emit": emit_allowed,
                "block_reason": block_reason,
            },
        )

    def _calculate_prediction(
        self,
        *,
        mode: ControlMode,
        error_meas_x: float,
        filtered_velocity_x: float,
        motion_confidence: float,
        reference_dt_ms: float,
        estimate_available: bool,
    ) -> PredictionResult:
        prediction_config = self.config.prediction
        if not prediction_config.enabled:
            return PredictionResult(
                reference_dt_ms=0.0,
                lead_frames=0.0,
                raw_offset_x=0.0,
                weighted_offset_x=0.0,
                safe_offset_x=0.0,
                allowed_cap_x=0.0,
                motion_confidence=0.0,
                allowed=False,
            )
        valid_dt = reference_dt_ms if isfinite(reference_dt_ms) and reference_dt_ms > 0.0 else 0.0
        allowed = bool(
            prediction_config.enabled
            and estimate_available
            and valid_dt > 0.0
            and prediction_config.lead_frames > 0.0
        )
        effective_confidence = _clamp(motion_confidence, 0.0, 1.0) if allowed else 0.0
        raw_offset_x = (
            filtered_velocity_x * valid_dt * prediction_config.lead_frames
            if prediction_config.enabled
            else 0.0
        )
        weighted_offset_x = raw_offset_x * effective_confidence
        mode_config = prediction_config.far if mode is ControlMode.FAR else prediction_config.near
        allowed_cap_x = min(
            mode_config.absolute_cap_px,
            mode_config.base_cap_px + mode_config.relative_cap * abs(error_meas_x),
        )
        safe_offset_x = _clamp(
            weighted_offset_x,
            -allowed_cap_x,
            allowed_cap_x,
        )
        return PredictionResult(
            reference_dt_ms=valid_dt,
            lead_frames=(prediction_config.lead_frames if prediction_config.enabled else 0.0),
            raw_offset_x=raw_offset_x,
            weighted_offset_x=weighted_offset_x,
            safe_offset_x=safe_offset_x,
            allowed_cap_x=allowed_cap_x,
            motion_confidence=effective_confidence,
            allowed=allowed,
        )

    def _blocked_decision(
        self,
        observation: DualPhaseAtanRobustPredictiveV2Observation,
        reason: str,
        *,
        frame_age_ms: float | None = None,
    ) -> ControlDecision:
        self.release_trigger()
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
                "mode": ControlMode.FAR.value,
                "history_position_count": self._velocity_x.history_position_count,
                "motion_confidence": 0.0,
                "trigger_active": observation.trigger_active,
                "executor_success": None,
                "will_emit": False,
                "block_reason": reason,
                "quantizer_residual_x": 0.0,
                "quantizer_residual_y": 0.0,
            },
        )


class _AxisQuantizer:
    def __init__(self) -> None:
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
        output = trunc(self.accumulator)
        self.accumulator -= output
        return int(output), direction_reset


def _atan_demand(
    full_error_counts: float,
    config: AtanModeConfig,
    scale_counts: float,
) -> float:
    if not isfinite(full_error_counts) or not isfinite(scale_counts) or scale_counts <= 0.0:
        return 0.0
    value = config.kp * scale_counts * atan(full_error_counts / scale_counts)
    return _clamp(
        value,
        -config.max_counts_per_update,
        config.max_counts_per_update,
    )


def _valid_observation_geometry(
    observation: DualPhaseAtanRobustPredictiveV2Observation,
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
    width = observation.observation_width
    height = observation.observation_height
    return (
        all(int(value) > 0 for value in dimensions)
        and all(isfinite(float(value)) for value in coordinates)
        and observation.bbox_x2 > observation.bbox_x1
        and observation.bbox_y2 > observation.bbox_y1
        and 0.0 <= observation.aim_x <= width
        and 0.0 <= observation.aim_y <= height
        and 0.0 <= observation.crosshair_x <= width
        and 0.0 <= observation.crosshair_y <= height
        and observation.bbox_x2 > 0.0
        and observation.bbox_y2 > 0.0
        and observation.bbox_x1 < width
        and observation.bbox_y1 < height
    )


def _observation_geometry_signature(
    observation: DualPhaseAtanRobustPredictiveV2Observation,
) -> tuple[int, ...]:
    return (
        observation.observation_width,
        observation.observation_height,
        observation.roi_left,
        observation.roi_top,
        observation.roi_width,
        observation.roi_height,
        observation.source_width,
        observation.source_height,
    )


def _crossed_center(previous_error: float, current_error: float) -> bool:
    return (
        isfinite(previous_error)
        and isfinite(current_error)
        and previous_error * current_error < 0.0
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _validate_config(config: DualPhaseAtanRobustPredictiveV2Config) -> None:
    if config.freshness_threshold_ms <= 0.0:
        raise ValueError("freshness_threshold_ms must be > 0")
    if not 0.0 < config.projection.fov_x_deg < 180.0:
        raise ValueError("fov_x_deg must be in (0, 180)")
    if config.projection.counts_per_360 <= 0.0:
        raise ValueError("counts_per_360 must be > 0")
    if not isfinite(config.mode.near_threshold_px) or config.mode.near_threshold_px < 0.0:
        raise ValueError("near_threshold_px must be finite and >= 0")
    prediction = config.prediction
    if not 0.0 <= prediction.lead_frames <= 10.0:
        raise ValueError("prediction lead_frames must be in [0, 10]")
    for mode_config in (prediction.far, prediction.near):
        if (
            min(
                mode_config.absolute_cap_px,
                mode_config.base_cap_px,
                mode_config.relative_cap,
            )
            < 0.0
        ):
            raise ValueError("prediction caps must be >= 0")
    if not isfinite(config.atan.scale_counts) or config.atan.scale_counts <= 0.0:
        raise ValueError("Atan scale must be finite and > 0")
    for mode_config in (config.atan.far, config.atan.near):
        if (
            mode_config.kp <= 0.0
            or mode_config.max_counts_per_update <= 0.0
            or mode_config.max_counts_per_update > MAX_ABS_MOUSE_MOVE_COUNT
        ):
            raise ValueError(
                "Atan Kp must be > 0 and output limit must fit the kmNet move range"
            )
