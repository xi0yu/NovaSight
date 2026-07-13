from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


ALGORITHM_ID = "dual_phase_atan_robust_predictive_v2"


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
class VelocityConfig:
    history_size: int = 4
    velocity_sample_count: int = 3
    smoothing_tau_ms: float = 30.0
    history_reset_gap_ms: float = 80.0
    spread_base_px_ms: float = 0.10
    spread_relative: float = 0.50
    change_base_px_ms: float = 0.15
    change_relative: float = 0.75


@dataclass(frozen=True, slots=True)
class PredictionModeConfig:
    absolute_cap_px: float
    base_cap_px: float
    relative_cap: float


def _default_far_prediction() -> PredictionModeConfig:
    return PredictionModeConfig(
        absolute_cap_px=8.0,
        base_cap_px=1.0,
        relative_cap=0.25,
    )


def _default_near_prediction() -> PredictionModeConfig:
    return PredictionModeConfig(
        absolute_cap_px=2.0,
        base_cap_px=0.5,
        relative_cap=0.15,
    )


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    coefficient: float = 1.0
    actuation_delay_ms: float = 5.0
    max_horizon_ms: float = 35.0
    far: PredictionModeConfig = field(default_factory=_default_far_prediction)
    near: PredictionModeConfig = field(default_factory=_default_near_prediction)
    enabled_x: bool = True
    enabled_y: bool = False


@dataclass(frozen=True, slots=True)
class AtanModeConfig:
    kp: float
    scale_counts: float
    max_counts_per_update: float


def _default_far_atan() -> AtanModeConfig:
    return AtanModeConfig(
        kp=0.35,
        scale_counts=256.0,
        max_counts_per_update=127.0,
    )


def _default_near_atan() -> AtanModeConfig:
    return AtanModeConfig(
        kp=0.15,
        scale_counts=256.0,
        max_counts_per_update=60.0,
    )


@dataclass(frozen=True, slots=True)
class AtanControllerConfig:
    far: AtanModeConfig = field(default_factory=_default_far_atan)
    near: AtanModeConfig = field(default_factory=_default_near_atan)


@dataclass(frozen=True, slots=True)
class DualPhaseAtanRobustPredictiveV2Config:
    freshness_threshold_ms: float = 55.0
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    mode: ModeSelectorConfig = field(default_factory=ModeSelectorConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    atan: AtanControllerConfig = field(default_factory=AtanControllerConfig)


@dataclass(frozen=True, slots=True)
class DualPhaseAtanRobustPredictiveV2Observation:
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
    track_rebuilt: bool = False


@dataclass(frozen=True, slots=True)
class PositionSample:
    aim_x: float
    capture_ts_ns: int


@dataclass(frozen=True, slots=True)
class VelocityEstimate:
    raw_velocities: tuple[float, float, float]
    median_velocity: float
    filtered_velocity: float
    spread: float
    motion_confidence: float
    measurement_dt_ms: float
    history_position_count: int
    history_quality: float
    spread_quality: float
    trend_quality: float
    detection_quality: float
    track_quality: float


@dataclass(frozen=True, slots=True)
class PredictionResult:
    horizon_ms: float
    raw_offset_x: float
    coefficient_offset_x: float
    weighted_offset_x: float
    safe_offset_x: float
    allowed_cap_x: float
    motion_confidence: float
    allowed: bool


@dataclass(frozen=True, slots=True)
class ControlDecision:
    dx: int
    dy: int
    emit_allowed: bool
    block_reason: str
    telemetry: dict[str, Any]
