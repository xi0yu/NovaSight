from __future__ import annotations

from dataclasses import MISSING, Field, asdict, dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from novasight.control.registry import DEFAULT_ACTIVE_ALGORITHM_ID, supported_algorithm_ids
from novasight.roi import ROI_SIZE_CHOICES, normalize_roi_size


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 5174


@dataclass
class SourceConfig:
    default: str = "null"
    target_fps: int = 60
    image_path: str = ""
    image_fps: int = 30


@dataclass
class ConsumerConfig:
    preview: bool = True
    inference: bool = True
    recording: bool = False
    recording_format: str = "csv"
    recording_path: str = ""


@dataclass
class RuntimeLimitsConfig:
    stream_fps: int = 30


@dataclass
class RuntimeBehaviorConfig:
    freshness_threshold_ms: float = 55.0
    drop_stale_batches: bool = True
    consume_latest_only: bool = True


@dataclass
class PowerSavingConfig:
    host_presence_enabled: bool = False
    target_host_id: str = ""
    heartbeat_timeout_s: float = 6.0
    offline_grace_s: float = 15.0
    auto_resume: bool = True


@dataclass
class RoiConfig:
    size: int = 640


@dataclass
class CrosshairConfig:
    enabled: bool = False
    use_for_control: bool = False
    search_size: int = 96
    sample_hz: int = 10
    sample_frames: int = 5
    confirm_duration_ms: float = 200.0
    max_age_ms: float = 300.0
    max_offset_px: float = 20.0
    min_similarity: float = 0.68
    max_step_px: float = 2.0


@dataclass
class InferenceConfig:
    enabled: bool = True
    backend: str = "deepstream_nvinfer"
    device: str = "cuda"
    require_gpu: bool = True
    allow_cpu_fallback: bool = False
    inference_input_deadline_ms: float = 55.0
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    input_source: str = "source.default"
    detection_class_profile: str = "default"
    detection_class_filter: str = "all"
    detection_class_priority: str = "1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    detection_class_profiles: dict[str, list[str]] = field(
        default_factory=lambda: {
            "default": [
                "0-敌人/身体",
                "1-头部",
                "2-队友",
                "3-小兵",
                "4-倒地",
                "5-靶场",
                "6-靶场头",
                "7-类别7",
                "8-类别8",
                "9-类别9",
                "10-类别10",
                "11-类别11",
                "12-类别12",
                "13-类别13",
                "14-类别14",
                "15-类别15",
            ]
        }
    )
    deepstream_io_mode: int = 2
    deepstream_batched_push_timeout_us: int = 0
    deepstream_parser_library: str = "build/deepstream-parser/libnovasight_parser.so"


@dataclass
class PreprocessConfig:
    backend: str = "cuda"
    input_format: str = "auto"
    output_dtype: str = "fp16"
    normalize: bool = True
    use_pinned_memory: bool = True
    h2d_async: bool = True


@dataclass
class CaptureConfig:
    device: str = "/dev/video0"
    backend: str = "deepstream_nvinfer"
    preference: str = "auto_high_fps"
    memory: str = "nvmm"
    latest_only: bool = True
    appsink_max_buffers: int = 1
    queue_leaky: str = "downstream"
    pixel_format: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0


@dataclass
class CalibrationConfig:
    profile_id: str = "default"
    profile_version: int = 1
    game_sensitivity_fingerprint: str = "unverified-default"


@dataclass
class AimRoleRatiosConfig:
    head: float = 0.22
    body: float = 0.22
    other: float = 0.22


@dataclass
class AimConfig:
    role_y_ratios: AimRoleRatiosConfig = field(default_factory=AimRoleRatiosConfig)
    class_roles: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class CalibratedAngularConfig:
    fov_x_deg: float = 105.0
    counts_per_360_x: float = 9980.0
    counts_per_360_y: float = 9980.0
    kp_x: float = 1.0
    kp_y: float = 1.0
    kd_x: float = 0.0
    kd_y: float = 0.0
    d_ema_alpha: float = 0.30
    max_angle_step_x_deg: float = 2.0
    max_angle_step_y_deg: float = 1.5


@dataclass
class UniversalSaturatedConfig:
    response_scale_x_px: float = 80.0
    response_scale_y_px: float = 60.0
    max_step_x_counts: float = 50.0
    max_step_y_counts: float = 40.0


@dataclass
class DualPhaseProjectionConfig:
    fov_x_deg: float = 105.0
    counts_per_360: float = 9980.0
    invert_y: bool = False


@dataclass
class DualPhaseRobustModeSelectorConfig:
    near_threshold_px: float = 12.0


@dataclass
class DualPhaseRobustVelocityConfig:
    history_size: int = 4
    velocity_sample_count: int = 3
    smoothing_frames: float = 3.0
    history_reset_gap_ms: float = 80.0
    spread_base_px_ms: float = 0.12
    spread_relative: float = 0.50
    change_base_px_ms: float = 0.20
    change_relative: float = 0.75


@dataclass
class DualPhaseRobustPredictionModeConfig:
    absolute_cap_px: float
    base_cap_px: float
    relative_cap: float


def _default_dual_phase_robust_far_prediction() -> DualPhaseRobustPredictionModeConfig:
    return DualPhaseRobustPredictionModeConfig(
        absolute_cap_px=10.0,
        base_cap_px=1.25,
        relative_cap=0.30,
    )


def _default_dual_phase_robust_near_prediction() -> DualPhaseRobustPredictionModeConfig:
    return DualPhaseRobustPredictionModeConfig(
        absolute_cap_px=3.0,
        base_cap_px=0.75,
        relative_cap=0.20,
    )


@dataclass
class DualPhaseRobustPredictionConfig:
    enabled: bool = True
    lead_frames: float = 1.0
    far: DualPhaseRobustPredictionModeConfig = field(
        default_factory=_default_dual_phase_robust_far_prediction
    )
    near: DualPhaseRobustPredictionModeConfig = field(
        default_factory=_default_dual_phase_robust_near_prediction
    )


@dataclass
class DualPhaseRobustAtanModeConfig:
    kp: float
    max_counts_per_update: float


def _default_dual_phase_robust_far_atan() -> DualPhaseRobustAtanModeConfig:
    return DualPhaseRobustAtanModeConfig(
        kp=0.45,
        max_counts_per_update=127.0,
    )


def _default_dual_phase_robust_near_atan() -> DualPhaseRobustAtanModeConfig:
    return DualPhaseRobustAtanModeConfig(
        kp=0.22,
        max_counts_per_update=72.0,
    )


@dataclass
class DualPhaseRobustAtanConfig:
    scale_counts: float = 256.0
    far: DualPhaseRobustAtanModeConfig = field(default_factory=_default_dual_phase_robust_far_atan)
    near: DualPhaseRobustAtanModeConfig = field(
        default_factory=_default_dual_phase_robust_near_atan
    )


@dataclass
class DualPhaseAtanRobustPredictiveV2Config:
    schema_version: int = 8
    freshness_threshold_ms: float = 55.0
    projection: DualPhaseProjectionConfig = field(default_factory=DualPhaseProjectionConfig)
    mode: DualPhaseRobustModeSelectorConfig = field(
        default_factory=DualPhaseRobustModeSelectorConfig
    )
    velocity: DualPhaseRobustVelocityConfig = field(default_factory=DualPhaseRobustVelocityConfig)
    prediction: DualPhaseRobustPredictionConfig = field(
        default_factory=DualPhaseRobustPredictionConfig
    )
    atan: DualPhaseRobustAtanConfig = field(default_factory=DualPhaseRobustAtanConfig)


@dataclass
class ControlAlgorithmConfigs:
    calibrated_angular: CalibratedAngularConfig = field(default_factory=CalibratedAngularConfig)
    universal_saturated: UniversalSaturatedConfig = field(default_factory=UniversalSaturatedConfig)
    dual_phase_atan_robust_predictive_v2: DualPhaseAtanRobustPredictiveV2Config = field(
        default_factory=DualPhaseAtanRobustPredictiveV2Config
    )


@dataclass
class SharedControlConfig:
    max_count_slew_x: float = 10.0
    max_count_slew_y: float = 8.0
    deadzone_x_px: float = 4.0
    deadzone_y_px: float = 4.0
    arrival_hysteresis_enabled: bool = True
    invert_y: bool = False
    trigger_activation_delay_ms: float = 0.0


@dataclass
class RecoilConfig:
    enabled: bool = False
    base_rate_counts_s: float = 0.0
    max_rate_counts_s: float = 0.0
    startup_ms: float = 35.0
    positive_deadzone_norm: float = 0.04
    negative_deadzone_norm: float = 0.04
    full_brake_error_norm: float = 0.12
    fast_add_gain_counts_s: float = 0.0
    max_fast_add_ratio: float = 0.30
    stale_threshold_ms: float = 55.0


@dataclass
class HumanizedMotionConfig:
    """Runtime switch for an optional humanized target-motion profile.

    The profile is deliberately separate from recoil and persisted control
    tuning.  When disabled, the existing controller path is unchanged.
    """
    enabled: bool = False
    active_profile: str = ""
    spatial_curve_enabled: bool = True
    side_scale: float = 1.0
    max_side_ratio: float = 0.10
    near_fade_start_px: float = 24.0
    micro_bypass_px: float = 3.0
    dynamic_rebase_ratio: float = 0.25
    minimum_jerk_fallback: bool = True
    terminal_feedback_gain: float = 0.35
    # The built-in path is usable without a trained profile.  Fitts timing
    # controls the duration while the side ratio controls a subtle curve, not
    # an absolute cursor displacement.
    builtin_fitts_a_ms: float = 35.0
    builtin_fitts_b_ms: float = 55.0
    builtin_side_ratio: float = 0.012

    def __post_init__(self) -> None:
        bounded = {
            "side_scale": (0.0, 4.0),
            "max_side_ratio": (0.0, 1.0),
            "near_fade_start_px": (0.0, 10000.0),
            "micro_bypass_px": (0.0, 1000.0),
            "dynamic_rebase_ratio": (0.05, 1.0),
            "terminal_feedback_gain": (0.05, 2.0),
            "builtin_fitts_a_ms": (0.0, 1000.0),
            "builtin_fitts_b_ms": (1.0, 1000.0),
            "builtin_side_ratio": (-0.15, 0.15),
        }
        for name, (lower, upper) in bounded.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < lower or value > upper:
                raise ValueError(f"humanized_motion.{name} must be in [{lower}, {upper}]")


@dataclass
class ControlConfig:
    active_algorithm: str = DEFAULT_ACTIVE_ALGORITHM_ID
    # Global runtime gate for mouse offset delivery.  Detection, tracking,
    # control calculation, and the kmNet connection stay alive when disabled.
    output_enabled: bool = True
    recoil: RecoilConfig = field(default_factory=RecoilConfig)
    target_fov_radius_px: float = 180.0
    target_switch_delay_ms: float = 50.0
    target_lock_enabled: bool = True
    target_sticky_bias: float = 0.25
    candidate_ratio_max_aspect: float = 6.0
    candidate_selection_class_weight: float = 0.55
    candidate_selection_distance_weight: float = 0.40
    tracker_max_match_distance: float = 1.5
    tracker_position_cost_weight: float = 0.75
    tracker_iou_cost_weight: float = 0.25
    tracker_max_missed_frames: int = 2
    target_switch_min_preference_advantage: float = 0.08
    target_switch_min_continuity_score: float = 0.70
    kalman_acceleration_noise: float = 1200.0
    kalman_measurement_noise_x: float = 16.0
    kalman_measurement_noise_y: float = 16.0
    kalman_max_predict_missing_ms: float = 80.0
    kalman_max_predict_steps: int = 5
    kalman_max_predict_dt_ms: float = 35.0
    kalman_max_position_sigma_px: float = 45.0
    kalman_max_covariance_trace: float = 5000.0
    kalman_nis_threshold: float = 9.21
    kalman_nis_hard_reject: float = 16.0
    kalman_min_identity_confidence: float = 0.70
    kalman_min_prediction_confidence: float = 0.35
    kalman_prediction_decay_tau_ms: float = 45.0
    aim: AimConfig = field(default_factory=AimConfig)
    algorithms: ControlAlgorithmConfigs = field(default_factory=ControlAlgorithmConfigs)
    shared: SharedControlConfig = field(default_factory=SharedControlConfig)
    humanized_motion: HumanizedMotionConfig = field(default_factory=HumanizedMotionConfig)
    configured_actuation_delay_s: float = 0.004
    scheduler_enabled: bool = True
    scheduler_step_counts_x: int = 8
    scheduler_step_counts_y: int = 8
    scheduler_interval_ms: float = 4.0
    trigger_mode: str = "always"

    # Transitional Python aliases keep internal callers and older extensions
    # working while serialized config uses the explicit algorithm namespace.
    @property
    def mode(self) -> str:
        return self.active_algorithm

    @mode.setter
    def mode(self, value: str) -> None:
        self.active_algorithm = str(value)

    @property
    def calibrated_angular(self) -> CalibratedAngularConfig:
        return self.algorithms.calibrated_angular

    @property
    def universal_saturated(self) -> UniversalSaturatedConfig:
        return self.algorithms.universal_saturated

    @property
    def dual_phase_atan_robust_predictive_v2(
        self,
    ) -> DualPhaseAtanRobustPredictiveV2Config:
        return self.algorithms.dual_phase_atan_robust_predictive_v2


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "logs"


@dataclass
class HardwareConfig:
    auto_connect: bool = True
    host: str = "192.168.2.188"
    port: int = 8888
    uuid: str = "12345678"
    monitor_port: int = 5001


@dataclass
class RuntimeConfig:
    web: WebConfig = field(default_factory=WebConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    consumers: ConsumerConfig = field(default_factory=ConsumerConfig)
    limits: RuntimeLimitsConfig = field(default_factory=RuntimeLimitsConfig)
    runtime: RuntimeBehaviorConfig = field(default_factory=RuntimeBehaviorConfig)
    power_saving: PowerSavingConfig = field(default_factory=PowerSavingConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    crosshair: CrosshairConfig = field(default_factory=CrosshairConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


T = TypeVar("T")


def _field_default(item: Field[Any]) -> Any:
    if item.default_factory is not MISSING:
        return item.default_factory()
    if item.default is not MISSING:
        return item.default
    return None


def _validate_leaf_value(key_name: str, value: Any, expected_type: type[Any]) -> None:
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"runtime config key '{key_name}' must be a list")
        if args and args[0] is str and not all(isinstance(item, str) for item in value):
            raise ValueError(f"runtime config key '{key_name}' must be a list of strings")
        return
    if expected_type is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"runtime config key '{key_name}' must be an int")
    elif expected_type is float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"runtime config key '{key_name}' must be a number")
    elif expected_type is str:
        if not isinstance(value, str):
            raise ValueError(f"runtime config key '{key_name}' must be a string")
    elif expected_type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"runtime config key '{key_name}' must be a boolean")


def _build_dataclass(
    cls: type[T],
    raw: dict[str, Any],
    section: str = "",
    base: T | None = None,
) -> T:
    items = {item.name: item for item in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown_keys = sorted(set(raw) - set(items), key=str)
    if unknown_keys:
        key_names = ", ".join(f"{section}.{key}" if section else str(key) for key in unknown_keys)
        raise ValueError(f"unknown config key(s): {key_names}")

    values: dict[str, Any] = {}
    for item in items.values():
        if item.name not in raw:
            continue
        value = raw[item.name]
        current = getattr(base, item.name) if base is not None else _field_default(item)
        if is_dataclass(current) and isinstance(value, dict):
            key_name = f"{section}.{item.name}" if section else item.name
            values[item.name] = _build_dataclass(
                type(current),
                value,
                key_name,
                base=current,
            )
        elif is_dataclass(current):
            key_name = f"{section}.{item.name}" if section else item.name
            raise ValueError(f"runtime config section '{key_name}' must be a mapping")
        else:
            key_name = f"{section}.{item.name}" if section else item.name
            _validate_leaf_value(key_name, value, type_hints[item.name])
            values[item.name] = value
    return replace(base, **values) if base is not None else cls(**values)


def _drop_legacy_runtime_keys(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    roi = normalized.get("roi")
    if isinstance(roi, dict):
        roi = dict(roi)
        # ROI is always centered. Keep accepting old files during migration,
        # but never carry the former mode/offset controls into runtime state.
        for key in ("mode", "offset_x", "offset_y"):
            roi.pop(key, None)
        normalized["roi"] = roi
    inference = normalized.get("inference")
    if isinstance(inference, dict):
        inference = dict(inference)
        legacy_backend = str(inference.get("backend", "")).lower()
        if legacy_backend in {
            "deepstream",
            "tensorrt",
            "nvmm_latest",
            "gst_cpu_latest",
            "onnxruntime",
            "legacy_latest",
            "deepstream_uncontrolled",
        }:
            inference["backend"] = "deepstream_nvinfer"
        for key in (
            "deepstream_manifest_path",
            "deepstream_config_path",
            "deepstream_tracker_config_path",
        ):
            inference.pop(key, None)
        normalized["inference"] = inference
    capture = normalized.get("capture")
    if isinstance(capture, dict):
        capture = dict(capture)
        memory = str(capture.get("memory", "")).lower()
        if memory in {"cpu", "system"}:
            capture["memory"] = "nvmm"
        legacy_capture_backend = str(capture.get("backend", "")).lower()
        if legacy_capture_backend in {
            "deepstream",
            "gst_cpu_latest",
            "nvmm_latest",
            "legacy_latest",
            "deepstream_uncontrolled",
        }:
            capture["backend"] = "deepstream_nvinfer"
            capture["memory"] = "nvmm"
        normalized["capture"] = capture
    if isinstance(inference, dict) and inference.get("backend") == "deepstream_nvinfer":
        capture = dict(normalized.get("capture") or {})
        capture["backend"] = "deepstream_nvinfer"
        capture["memory"] = "nvmm"
        normalized["capture"] = capture
        preprocess = dict(normalized.get("preprocess") or {})
        preprocess["backend"] = "cuda"
        normalized["preprocess"] = preprocess
    hardware = normalized.get("hardware")
    if isinstance(hardware, dict):
        hardware = dict(hardware)
        # The preceding kmNet executor accepted this key but forced it to False.
        hardware.pop("flip_dy", None)
        for key in (
            "kind",
            "serial_port",
            "heartbeat_timeout_ms",
            "min_effective_move_counts_x",
            "min_effective_move_counts_y",
        ):
            hardware.pop(key, None)
        normalized["hardware"] = hardware
    normalized.pop("executor", None)
    control = normalized.get("control")
    calibration = normalized.get("calibration")
    if isinstance(control, dict):
        control = dict(control)
        for key in (
            "output_mode",
            "lost_target_timeout_ms",
            "tracker_confirm_frames",
            "tracker_matching_distance_px",
            "tracker_ambiguity_margin",
            "tracker_delete_timeout_ms",
            "tracker_match_threshold",
            "tracker_mahalanobis_gate",
            "class_priority_quality_margin",
            "candidate_selection_quality_weight",
            "candidate_quality_confidence_weight",
            "candidate_quality_area_weight",
        ):
            control.pop(key, None)
    if isinstance(calibration, dict):
        calibration = dict(calibration)
        _migrate_legacy_axis_signs(calibration)
        if control is None and any(
            key in calibration
            for key in ("fov_x_deg", "counts_per_360_x", "counts_per_360_y", "invert_y")
        ):
            control = {}
    if isinstance(control, dict):
        if calibration is None:
            calibration = {}
        if isinstance(calibration, dict):
            _migrate_legacy_control_calibration(control, calibration)
        _migrate_legacy_mouse_control(control, normalized)
        if isinstance(calibration, dict):
            _migrate_dual_control_modes(control, calibration)
        _migrate_control_algorithm_namespaces(control)
        _migrate_shared_aim_config(control)
        _migrate_fixed_recoil_config(control)
        normalized["control"] = control
    if isinstance(calibration, dict):
        normalized["calibration"] = calibration
    return normalized


def _migrate_legacy_control_calibration(
    control: dict[str, Any],
    calibration: dict[str, Any],
) -> None:
    legacy_fov = control.pop("experimental_angle_fov_x_deg", None)
    if legacy_fov is not None and "fov_x_deg" not in calibration:
        calibration["fov_x_deg"] = legacy_fov

    legacy_counts = control.pop("experimental_angle_counts_per_360", None)
    if legacy_counts is not None:
        calibration.setdefault("counts_per_360_x", legacy_counts)
        calibration.setdefault("counts_per_360_y", legacy_counts)

    legacy_sign_x = control.pop("experimental_angle_sign_x", None)
    legacy_sign_y = control.pop("experimental_angle_sign_y", None)
    if legacy_sign_x is not None:
        calibration.setdefault("axis_sign_x", legacy_sign_x)
    if legacy_sign_y is not None:
        calibration.setdefault("axis_sign_y", legacy_sign_y)


def _migrate_legacy_axis_signs(calibration: dict[str, Any]) -> None:
    legacy_sign_x = calibration.pop("axis_sign_x", None)
    legacy_sign_y = calibration.pop("axis_sign_y", None)
    if legacy_sign_x is not None:
        sign_x = _legacy_axis_sign(legacy_sign_x, "calibration.axis_sign_x")
        if sign_x != 1:
            raise ValueError(
                "legacy config key 'calibration.axis_sign_x=-1' cannot be migrated: "
                "the single mouse-control route fixes positive X counts to positive X error"
            )
    if legacy_sign_y is not None:
        sign_y = _legacy_axis_sign(legacy_sign_y, "calibration.axis_sign_y")
        if "invert_y" not in calibration:
            calibration["invert_y"] = sign_y < 0


def _migrate_legacy_mouse_control(
    control: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    legacy_schema = any(
        key in _LEGACY_MOUSE_CONTROL_MARKERS or key.startswith("experimental_angle_")
        for key in control
    )
    legacy_strategy = control.get("strategy")
    if legacy_strategy is not None and legacy_strategy != "experimental_angle_pid":
        raise ValueError("legacy config key 'control.strategy' must be experimental_angle_pid")
    if legacy_schema:
        control.setdefault("mode", "calibrated_angular")

    legacy_aim_ratio = control.pop("aim_ratio", None)
    if legacy_aim_ratio is not None:
        aim = control.get("aim")
        if aim is None:
            aim = {}
        if isinstance(aim, dict):
            aim = dict(aim)
            if "y_ratio" not in aim:
                ratio = _legacy_number(legacy_aim_ratio, "control.aim_ratio") / 100.0
                aim["y_ratio"] = round(max(0.0, min(1.0, ratio)), 2)
            control["aim"] = aim

    legacy_delay_ms = control.pop("configured_extra_prediction_delay_ms", None)
    legacy_estimated_delay_ms = control.pop("latency_estimated_actuation_delay_ms", None)
    if "configured_actuation_delay_s" not in control:
        delay_ms = legacy_delay_ms if legacy_delay_ms is not None else legacy_estimated_delay_ms
        if delay_ms is not None:
            control["configured_actuation_delay_s"] = (
                _legacy_number(delay_ms, "control.configured_extra_prediction_delay_ms") / 1000.0
            )

    control.pop("latency_compensation_enabled", None)
    control.pop("latency_compensation_scale", None)
    control.pop("prediction_strength", None)
    control.pop("prediction_x_enabled", None)
    control.pop("prediction_y_enabled", None)
    _move_legacy_number(control, "experimental_angle_kp_x", "kp_x")
    _move_legacy_number(control, "experimental_angle_kp_y", "kp_y")

    legacy_kd = control.pop("experimental_angle_kd", None)
    if legacy_kd is not None:
        control.setdefault("kd_x", legacy_kd)
        control.setdefault("kd_y", legacy_kd)
    _move_legacy_number(
        control,
        "experimental_angle_derivative_filter",
        "d_ema_alpha",
    )

    legacy_deadzone = control.pop("experimental_angle_deadzone_px", None)
    if legacy_deadzone is not None:
        control.setdefault("deadzone_px_x", legacy_deadzone)
        control.setdefault("deadzone_px_y", legacy_deadzone)

    legacy_max_angle_deg = control.pop("experimental_angle_max_control_angle_deg", None)
    if legacy_max_angle_deg is not None:
        max_angle_rad = math.radians(
            _legacy_number(
                legacy_max_angle_deg,
                "control.experimental_angle_max_control_angle_deg",
            )
        )
        control.setdefault("max_output_rad_x", max_angle_rad)
        control.setdefault("max_output_rad_y", max_angle_rad)

    _move_legacy_number(control, "command_interval_ms", "scheduler_interval_ms")
    _move_legacy_number(control, "scheduler_max_step_x", "scheduler_step_counts_x")
    _move_legacy_number(control, "scheduler_max_step_y", "scheduler_step_counts_y")
    control.pop("tracker_missing_timeout_ms", None)

    legacy_stale_ms = control.pop("latency_reject_if_age_exceeds_ms", None)
    if legacy_stale_ms is not None:
        runtime = normalized.get("runtime")
        if runtime is None:
            runtime = {}
        if isinstance(runtime, dict):
            runtime = dict(runtime)
            runtime.setdefault("freshness_threshold_ms", legacy_stale_ms)
            normalized["runtime"] = runtime

    control.pop("min_confidence", None)

    for key in tuple(control):
        if key.startswith("experimental_angle_"):
            control.pop(key, None)
    for key in _REMOVED_LEGACY_CONTROL_KEYS:
        control.pop(key, None)


def _migrate_dual_control_modes(
    control: dict[str, Any],
    calibration: dict[str, Any],
) -> None:
    if isinstance(control.get("algorithms"), dict):
        return
    calibrated = control.get("calibrated_angular")
    if calibrated is None:
        calibrated = {}
    if not isinstance(calibrated, dict):
        return
    calibrated = dict(calibrated)
    universal = control.get("universal_saturated")
    if universal is None:
        universal = {}
    if not isinstance(universal, dict):
        return
    universal = dict(universal)
    old_universal_defaults = {
        "response_scale_x_px": 160.0,
        "response_scale_y_px": 120.0,
        "max_step_x_counts": 30.0,
        "max_step_y_counts": 24.0,
    }
    if all(
        isinstance(universal.get(key), (int, float))
        and not isinstance(universal.get(key), bool)
        and float(universal[key]) == expected
        for key, expected in old_universal_defaults.items()
    ):
        universal.update(
            {
                "response_scale_x_px": 80.0,
                "response_scale_y_px": 60.0,
                "max_step_x_counts": 50.0,
                "max_step_y_counts": 40.0,
            }
        )
    shared = control.get("shared")
    if shared is None:
        shared = {}
    if not isinstance(shared, dict):
        return
    shared = dict(shared)
    for key in ("scheduler_step_counts_x", "scheduler_step_counts_y"):
        value = control.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            control[key] = 8 if value == 32 else min(20, max(1, value))
    if "arrival_hysteresis_enabled" not in shared:
        legacy_x = shared.get("deadzone_x_px")
        legacy_y = shared.get("deadzone_y_px")
        if legacy_x == 0.0 and legacy_y == 0.0:
            shared["deadzone_x_px"] = 4.0
            shared["deadzone_y_px"] = 4.0
        shared["arrival_hysteresis_enabled"] = True

    legacy_calibrated_fields = {
        "kp_x": "kp_x",
        "kp_y": "kp_y",
        "kd_x": "kd_x",
        "kd_y": "kd_y",
        "d_ema_alpha": "d_ema_alpha",
    }
    legacy_mode_detected = any(key in control for key in legacy_calibrated_fields) or any(
        key in calibration for key in ("fov_x_deg", "counts_per_360_x", "counts_per_360_y")
    )
    for old_key, new_key in legacy_calibrated_fields.items():
        value = control.pop(old_key, None)
        if value is not None:
            calibrated.setdefault(new_key, value)

    for key in ("fov_x_deg", "counts_per_360_x", "counts_per_360_y"):
        value = calibration.pop(key, None)
        if value is not None:
            calibrated.setdefault(key, value)

    for axis in ("x", "y"):
        old_key = f"max_output_rad_{axis}"
        value = control.pop(old_key, None)
        if value is not None:
            numeric = _legacy_number(value, f"control.{old_key}")
            calibrated.setdefault(f"max_angle_step_{axis}_deg", math.degrees(numeric))
        control.pop(f"max_output_rate_rad_s_{axis}", None)

        deadzone = control.pop(f"deadzone_px_{axis}", None)
        if deadzone is not None:
            shared.setdefault(f"deadzone_{axis}_px", deadzone)

    invert_y = calibration.pop("invert_y", None)
    if invert_y is not None:
        shared.setdefault("invert_y", invert_y)
    calibration.pop("fov_semantics", None)
    calibration.pop("projection_profile", None)

    if legacy_mode_detected:
        control.setdefault("mode", "calibrated_angular")
    control["calibrated_angular"] = calibrated
    control["universal_saturated"] = universal
    control["shared"] = shared


def _migrate_control_algorithm_namespaces(control: dict[str, Any]) -> None:
    raw_algorithms = control.get("algorithms")
    if raw_algorithms is not None and not isinstance(raw_algorithms, dict):
        return
    algorithms = dict(raw_algorithms or {})
    for algorithm_id in (
        "calibrated_angular",
        "universal_saturated",
        "dual_phase_atan_robust_predictive_v2",
    ):
        legacy_config = control.pop(algorithm_id, None)
        if legacy_config is not None:
            algorithms.setdefault(algorithm_id, legacy_config)

    removed_algorithm_ids = (
        "ttbox_pid_atan",
        "dual_phase_atan_predictive_v1",
    )
    for removed_algorithm_id in removed_algorithm_ids:
        control.pop(removed_algorithm_id, None)
        algorithms.pop(removed_algorithm_id, None)

    _migrate_dual_phase_robust_v2_namespace(algorithms)

    active_algorithm = control.pop("active_algorithm", None)
    legacy_mode = control.pop("mode", None)
    selected_algorithm = str(
        active_algorithm
        if active_algorithm is not None
        else legacy_mode
        if legacy_mode is not None
        else DEFAULT_ACTIVE_ALGORITHM_ID
    )
    if selected_algorithm in removed_algorithm_ids:
        selected_algorithm = "dual_phase_atan_robust_predictive_v2"
    control["active_algorithm"] = selected_algorithm
    control["algorithms"] = algorithms


def _migrate_dual_phase_robust_v2_namespace(
    algorithms: dict[str, Any],
) -> None:
    raw_config = algorithms.get("dual_phase_atan_robust_predictive_v2")
    if not isinstance(raw_config, dict):
        return

    config = dict(raw_config)
    mode = config.get("mode")
    if isinstance(mode, dict):
        mode = dict(mode)
        threshold = mode.get("near_threshold_px", mode.get("near_enter_min_px", 12.0))
        config["mode"] = {"near_threshold_px": threshold}

    atan = config.get("atan")
    if isinstance(atan, dict):
        atan = dict(atan)
        far = dict(atan.get("far", {})) if isinstance(atan.get("far"), dict) else {}
        near = dict(atan.get("near", {})) if isinstance(atan.get("near"), dict) else {}
        shared_scale = atan.get(
            "scale_counts",
            far.get("scale_counts", near.get("scale_counts", 256.0)),
        )
        far.pop("scale_counts", None)
        near.pop("scale_counts", None)
        atan["scale_counts"] = shared_scale
        atan["far"] = far
        atan["near"] = near
        config["atan"] = atan

    schema_version = int(config.get("schema_version", 2))
    if schema_version <= 3:
        velocity = dict(config.get("velocity") or {})
        legacy_tau_ms = velocity.pop("smoothing_tau_ms", None)
        if "smoothing_frames" not in velocity:
            velocity["smoothing_frames"] = (
                max(0.1, min(20.0, float(legacy_tau_ms) / (1000.0 / 120.0)))
                if isinstance(legacy_tau_ms, (int, float)) and not isinstance(legacy_tau_ms, bool)
                else 3.0
            )
        config["velocity"] = velocity

        prediction = dict(config.get("prediction") or {})
        legacy_enabled_x = prediction.pop("enabled_x", None)
        prediction.pop("enabled_y", None)
        legacy_coefficient = prediction.pop("coefficient", None)
        prediction.pop("max_horizon_ms", None)
        prediction.pop("max_lead_frames", None)
        prediction.pop("actuation_delay_ms", None)
        if "lead_frames" not in prediction:
            prediction["lead_frames"] = (
                0.0
                if legacy_enabled_x is False
                else legacy_coefficient
                if isinstance(legacy_coefficient, (int, float))
                and not isinstance(legacy_coefficient, bool)
                else 1.0
            )
        config["prediction"] = prediction
        config["schema_version"] = 4

    if int(config.get("schema_version", 4)) <= 4:
        config["schema_version"] = 5

    if int(config.get("schema_version", 5)) <= 5:
        config["schema_version"] = 6

    if int(config.get("schema_version", 6)) <= 6:
        config["schema_version"] = 7

    if int(config.get("schema_version", 7)) <= 7:
        atan = dict(config.get("atan") or {})
        far = dict(atan.get("far") or {})
        near = dict(atan.get("near") or {})
        aggressive_profile = (
            float(atan.get("scale_counts", 256.0)) == 1024.0
            and float(far.get("kp", 0.45)) == 0.90
            and float(far.get("max_counts_per_update", 127.0)) == 600.0
            and float(near.get("kp", 0.22)) == 0.30
            and float(near.get("max_counts_per_update", 72.0)) == 120.0
        )
        if aggressive_profile:
            atan["scale_counts"] = 256.0
            far["kp"] = 0.45
            far["max_counts_per_update"] = 127.0
            near["kp"] = 0.22
            near["max_counts_per_update"] = 72.0
        atan["far"] = far
        atan["near"] = near
        config["atan"] = atan
        config["schema_version"] = 8

    prediction = dict(config.get("prediction") or {})
    if "enabled" not in prediction:
        lead_frames = float(prediction.get("lead_frames", 1.0))
        prediction["enabled"] = lead_frames > 0.0
        if lead_frames <= 0.0:
            prediction["lead_frames"] = 1.0
    config["prediction"] = prediction
    algorithms["dual_phase_atan_robust_predictive_v2"] = config


def _migrate_shared_aim_config(control: dict[str, Any]) -> None:
    algorithms = control.get("algorithms")
    legacy_algorithm_ratio: Any = None
    if isinstance(algorithms, dict):
        robust = algorithms.get("dual_phase_atan_robust_predictive_v2")
        if isinstance(robust, dict):
            robust = dict(robust)
            legacy_aim = robust.pop("aim", None)
            if isinstance(legacy_aim, dict):
                legacy_algorithm_ratio = legacy_aim.get("y_ratio")
            algorithms = dict(algorithms)
            algorithms["dual_phase_atan_robust_predictive_v2"] = robust
            control["algorithms"] = algorithms

    aim = dict(control.get("aim") or {})
    legacy_shared_ratio = aim.pop("y_ratio", None)
    # Per-class ratios belonged to the old model. They cannot be mapped to a
    # semantic role safely, so migration intentionally falls back to `other`.
    aim.pop("class_y_ratios", None)
    if "role_y_ratios" not in aim:
        inherited_ratio = (
            legacy_shared_ratio
            if legacy_shared_ratio is not None
            else legacy_algorithm_ratio
            if legacy_algorithm_ratio is not None
            else 0.22
        )
        aim["role_y_ratios"] = {
            "head": inherited_ratio,
            "body": inherited_ratio,
            "other": inherited_ratio,
        }
    control["aim"] = aim


def _migrate_fixed_recoil_config(control: dict[str, Any]) -> None:
    shared = control.get("shared")
    if not isinstance(shared, dict):
        shared = {}
    shared = dict(shared)
    recoil = dict(control.get("recoil") or {})
    # Retire the old per-observation implementation without carrying any of
    # its semantics into the independent time-rate controller.
    for key in tuple(shared):
        if key.startswith("recoil_"):
            shared.pop(key, None)
    control["shared"] = shared
    control["recoil"] = recoil


_REMOVED_LEGACY_CONTROL_KEYS = frozenset(
    {
        "max_abs_dx",
        "max_abs_dy",
        "fov_ratio",
        "target_lost_grace_frames",
        "target_switch_confirm_frames",
        "kalman_enabled",
        "aim_horizontal_percent",
        "aim_offset_x_px",
        "aim_offset_y_px",
        "aim_ema_enabled",
        "aim_ema_alpha",
        "aim_max_anchor_jump_ratio",
        "latency_max_compensation_ms",
        "latency_max_compensation_px",
        "latency_min_velocity_px_s",
        "latency_max_velocity_px_s",
        "latency_min_velocity_measurements",
        "latency_min_velocity_confidence",
        "strategy",
        "scheduler_command_ttl_ms",
        "scheduler_predicted_command_ttl_ms",
        "scheduler_cancel_on_new_frame",
        "scheduler_cancel_on_direction_change",
        "scheduler_cancel_on_track_change",
        "scheduler_queue_hard_limit",
        "scheduler_device_error_cooldown_ms",
        "move_kind",
        "move_ms",
        "trace_ms",
        "bezier_curvature",
    }
)

_LEGACY_MOUSE_CONTROL_MARKERS = _REMOVED_LEGACY_CONTROL_KEYS | frozenset(
    {
        "aim_ratio",
        "configured_extra_prediction_delay_ms",
        "latency_estimated_actuation_delay_ms",
        "latency_compensation_enabled",
        "latency_compensation_scale",
        "latency_reject_if_age_exceeds_ms",
        "command_interval_ms",
        "scheduler_max_step_x",
        "scheduler_max_step_y",
        "tracker_missing_timeout_ms",
    }
)


def _move_legacy_number(
    values: dict[str, Any],
    old_key: str,
    new_key: str,
) -> None:
    legacy_value = values.pop(old_key, None)
    if legacy_value is not None and new_key not in values:
        values[new_key] = legacy_value


def _legacy_axis_sign(value: Any, key: str) -> int:
    numeric = _legacy_number(value, key)
    if numeric not in {-1.0, 1.0}:
        raise ValueError(f"legacy config key '{key}' must be -1 or 1")
    return int(numeric)


def _legacy_number(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"legacy config key '{key}' must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"legacy config key '{key}' must be finite")
    return numeric


def _validate_dual_phase_robust_v2_algorithm(
    cfg: DualPhaseAtanRobustPredictiveV2Config,
) -> None:
    prefix = "control.algorithms.dual_phase_atan_robust_predictive_v2"

    def finite(name: str, value: float) -> float:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"runtime config key '{prefix}.{name}' must be finite")
        return numeric

    if int(cfg.schema_version) != 8:
        raise ValueError(f"runtime config key '{prefix}.schema_version' must be 8")
    freshness_ms = finite("freshness_threshold_ms", cfg.freshness_threshold_ms)
    if freshness_ms <= 0.0:
        raise ValueError(f"runtime config key '{prefix}.freshness_threshold_ms' must be > 0")

    fov_x_deg = finite("projection.fov_x_deg", cfg.projection.fov_x_deg)
    if not 30.0 <= fov_x_deg <= 179.0:
        raise ValueError(f"runtime config key '{prefix}.projection.fov_x_deg' must be in [30, 179]")
    if finite("projection.counts_per_360", cfg.projection.counts_per_360) <= 0.0:
        raise ValueError(f"runtime config key '{prefix}.projection.counts_per_360' must be > 0")

    near_threshold = finite("mode.near_threshold_px", cfg.mode.near_threshold_px)
    if near_threshold < 0.0:
        raise ValueError(f"runtime config key '{prefix}.mode.near_threshold_px' must be >= 0")

    velocity = cfg.velocity
    if velocity.history_size != 4 or velocity.velocity_sample_count != 3:
        raise ValueError(
            f"runtime config key '{prefix}.velocity' requires "
            "history_size=4 and velocity_sample_count=3"
        )
    for key in (
        "smoothing_frames",
        "history_reset_gap_ms",
        "spread_base_px_ms",
        "change_base_px_ms",
    ):
        if finite(f"velocity.{key}", getattr(velocity, key)) <= 0.0:
            raise ValueError(f"runtime config key '{prefix}.velocity.{key}' must be > 0")
    for key in ("spread_relative", "change_relative"):
        if finite(f"velocity.{key}", getattr(velocity, key)) < 0.0:
            raise ValueError(f"runtime config key '{prefix}.velocity.{key}' must be >= 0")

    prediction = cfg.prediction
    lead_frames = finite("prediction.lead_frames", prediction.lead_frames)
    if not 0.0 <= lead_frames <= 10.0:
        raise ValueError(f"runtime config key '{prefix}.prediction.lead_frames' must be in [0, 10]")
    for mode_name, mode_cfg in (("far", prediction.far), ("near", prediction.near)):
        for key in ("absolute_cap_px", "base_cap_px", "relative_cap"):
            if finite(f"prediction.{mode_name}.{key}", getattr(mode_cfg, key)) < 0.0:
                raise ValueError(
                    f"runtime config key '{prefix}.prediction.{mode_name}.{key}' must be >= 0"
                )
    if (
        prediction.near.absolute_cap_px > prediction.far.absolute_cap_px
        or prediction.near.base_cap_px > prediction.far.base_cap_px
        or prediction.near.relative_cap > prediction.far.relative_cap
    ):
        raise ValueError(
            f"runtime config key '{prefix}.prediction' requires NEAR limits <= FAR limits"
        )

    far = cfg.atan.far
    near = cfg.atan.near
    if finite("atan.scale_counts", cfg.atan.scale_counts) <= 0.0:
        raise ValueError(f"runtime config key '{prefix}.atan.scale_counts' must be > 0")
    far_kp = finite("atan.far.kp", far.kp)
    near_kp = finite("atan.near.kp", near.kp)
    if not 0.0 < near_kp < far_kp < 1.0:
        raise ValueError(f"runtime config key '{prefix}.atan' requires 0 < near.kp < far.kp < 1")
    for mode_name, mode_cfg in (("far", far), ("near", near)):
        if (
            finite(
                f"atan.{mode_name}.max_counts_per_update",
                mode_cfg.max_counts_per_update,
            )
            <= 0.0
        ):
            raise ValueError(
                f"runtime config key '{prefix}.atan.{mode_name}.max_counts_per_update' must be > 0"
            )
        if mode_cfg.max_counts_per_update > 32_767.0:
            raise ValueError(
                f"runtime config key '{prefix}.atan.{mode_name}.max_counts_per_update' "
                "must be <= 32767 for one kmNet move command"
            )
    if near.max_counts_per_update > far.max_counts_per_update:
        raise ValueError(
            f"runtime config key '{prefix}.atan' requires "
            "near.max_counts_per_update <= far.max_counts_per_update"
        )


def _validate_runtime_rules(cfg: RuntimeConfig) -> None:
    if cfg.source.default not in {"null", "capture", "image"} and not cfg.source.default.startswith(
        "image:"
    ):
        raise ValueError(
            "runtime config key 'source.default' must be one of null, capture, image, or image:<path>"
        )
    if cfg.source.image_fps not in {1, 5, 15, 30, 60}:
        raise ValueError("runtime config key 'source.image_fps' must be one of 1, 5, 15, 30, 60")
    if cfg.capture.backend != "deepstream_nvinfer":
        raise ValueError(
            "runtime config key 'capture.backend' must be deepstream_nvinfer; "
            "CPU latest and NVMM latest data paths were removed"
        )
    if cfg.capture.memory != "nvmm":
        raise ValueError(
            "runtime config key 'capture.memory' must be nvmm for deepstream_nvinfer"
        )
    if not cfg.capture.latest_only:
        raise ValueError("runtime config key 'capture.latest_only' must be true")
    if cfg.capture.appsink_max_buffers != 1:
        raise ValueError("runtime config key 'capture.appsink_max_buffers' must be 1")
    if cfg.capture.queue_leaky != "downstream":
        raise ValueError("runtime config key 'capture.queue_leaky' must be downstream")
    if cfg.limits.stream_fps not in {15, 30}:
        raise ValueError("runtime config key 'limits.stream_fps' must be one of 15 or 30")
    if cfg.runtime.freshness_threshold_ms < 0:
        raise ValueError("runtime config key 'runtime.freshness_threshold_ms' must be >= 0")
    if not cfg.runtime.drop_stale_batches:
        raise ValueError("runtime config key 'runtime.drop_stale_batches' must be true")
    if not cfg.runtime.consume_latest_only:
        raise ValueError("runtime config key 'runtime.consume_latest_only' must be true")
    if (
        not math.isfinite(cfg.power_saving.heartbeat_timeout_s)
        or not 0 < cfg.power_saving.heartbeat_timeout_s <= 120
    ):
        raise ValueError(
            "runtime config key 'power_saving.heartbeat_timeout_s' must be finite "
            "and in (0, 120]"
        )
    if cfg.power_saving.host_presence_enabled and not cfg.power_saving.target_host_id.strip():
        raise ValueError(
            "runtime config key 'power_saving.target_host_id' must be non-empty "
            "when host presence is enabled"
        )
    if (
        not math.isfinite(cfg.power_saving.offline_grace_s)
        or not 0 <= cfg.power_saving.offline_grace_s <= 600
    ):
        raise ValueError(
            "runtime config key 'power_saving.offline_grace_s' must be finite "
            "and in [0, 600]"
        )
    if cfg.consumers.recording_format not in {"csv", "parquet"}:
        raise ValueError("runtime config key 'consumers.recording_format' must be csv or parquet")
    try:
        cfg.roi.size = normalize_roi_size(cfg.roi.size)
    except ValueError as exc:
        allowed = ", ".join(str(size) for size in ROI_SIZE_CHOICES)
        raise ValueError(f"unsupported ROI size: {cfg.roi.size}; must be one of {allowed}") from exc
    if cfg.crosshair.search_size < 32 or cfg.crosshair.search_size > cfg.roi.size:
        raise ValueError(
            "runtime config key 'crosshair.search_size' must be >= 32 and <= roi.size"
        )
    if cfg.crosshair.search_size % 2 != 0:
        raise ValueError("runtime config key 'crosshair.search_size' must be even")
    if cfg.crosshair.sample_hz < 1 or cfg.crosshair.sample_hz > 30:
        raise ValueError("runtime config key 'crosshair.sample_hz' must be in [1, 30]")
    if cfg.crosshair.sample_frames < 3 or cfg.crosshair.sample_frames > 15:
        raise ValueError("runtime config key 'crosshair.sample_frames' must be in [3, 15]")
    crosshair_bounds = {
        "confirm_duration_ms": (0.0, 2000.0),
        "max_age_ms": (50.0, 2000.0),
        "max_offset_px": (1.0, 64.0),
        "min_similarity": (0.0, 1.0),
        "max_step_px": (0.1, 20.0),
    }
    for key, (minimum, maximum) in crosshair_bounds.items():
        value = float(getattr(cfg.crosshair, key))
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(
                f"runtime config key 'crosshair.{key}' must be >= {minimum} and <= {maximum}"
            )
    if cfg.preprocess.backend != "cuda":
        raise ValueError(
            "runtime config key 'preprocess.backend' must be cuda for deepstream_nvinfer"
        )
    if cfg.preprocess.input_format != "auto":
        raise ValueError("runtime config key 'preprocess.input_format' must be auto")
    if cfg.preprocess.output_dtype not in {"fp16", "fp32", "float16", "float32"}:
        raise ValueError("runtime config key 'preprocess.output_dtype' must be fp16 or fp32")
    if cfg.inference.backend != "deepstream_nvinfer":
        raise ValueError(
            "runtime config key 'inference.backend' must be deepstream_nvinfer; "
            "custom TensorRT runtime data paths were removed"
        )
    if cfg.inference.device != "cuda":
        raise ValueError("runtime config key 'inference.device' must be cuda")
    if not cfg.inference.require_gpu:
        raise ValueError("runtime config key 'inference.require_gpu' must be true")
    if cfg.inference.allow_cpu_fallback:
        raise ValueError("runtime config key 'inference.allow_cpu_fallback' must be false")
    if cfg.inference.inference_input_deadline_ms < 0:
        raise ValueError("runtime config key 'inference.inference_input_deadline_ms' must be >= 0")
    if cfg.inference.confidence_threshold < 0 or cfg.inference.confidence_threshold > 1:
        raise ValueError(
            "runtime config key 'inference.confidence_threshold' must be >= 0 and <= 1"
        )
    if cfg.inference.nms_threshold < 0 or cfg.inference.nms_threshold > 1:
        raise ValueError("runtime config key 'inference.nms_threshold' must be >= 0 and <= 1")
    if cfg.inference.deepstream_io_mode < 0:
        raise ValueError("runtime config key 'inference.deepstream_io_mode' must be >= 0")
    if cfg.inference.deepstream_batched_push_timeout_us < 0:
        raise ValueError(
            "runtime config key 'inference.deepstream_batched_push_timeout_us' must be >= 0"
        )
    if not cfg.inference.deepstream_parser_library.strip():
        raise ValueError(
            "runtime config key 'inference.deepstream_parser_library' must be non-empty"
        )
    if cfg.capture.backend != "deepstream_nvinfer" or cfg.capture.memory != "nvmm":
        raise ValueError(
            "deepstream_nvinfer inference requires capture.backend=deepstream_nvinfer "
            "and capture.memory=nvmm"
        )
    if cfg.inference.input_source != "source.default":
        raise ValueError("runtime config key 'inference.input_source' must be source.default")
    if cfg.inference.detection_class_filter not in {"all", "none"}:
        _parse_detection_class_filter(cfg.inference.detection_class_filter)
    _parse_class_priority(cfg.inference.detection_class_priority)
    if cfg.inference.detection_class_profile not in cfg.inference.detection_class_profiles:
        raise ValueError(
            "runtime config key 'inference.detection_class_profile' must exist in detection_class_profiles"
        )
    if not cfg.calibration.profile_id.strip():
        raise ValueError("runtime config key 'calibration.profile_id' must be non-empty")
    if cfg.calibration.profile_version < 1:
        raise ValueError("runtime config key 'calibration.profile_version' must be >= 1")
    if not cfg.calibration.game_sensitivity_fingerprint.strip():
        raise ValueError(
            "runtime config key 'calibration.game_sensitivity_fingerprint' must be non-empty"
        )
    if cfg.control.active_algorithm not in supported_algorithm_ids():
        raise ValueError(
            "runtime config key 'control.active_algorithm' must select a configured algorithm"
        )
    for role_name in ("head", "body", "other"):
        ratio = float(getattr(cfg.control.aim.role_y_ratios, role_name))
        if not math.isfinite(ratio):
            raise ValueError(
                f"runtime config key 'control.aim.role_y_ratios.{role_name}' must be finite"
            )
        setattr(
            cfg.control.aim.role_y_ratios,
            role_name,
            round(max(0.0, min(1.0, ratio)), 2),
        )
    normalized_class_roles: dict[str, dict[str, str]] = {}
    for profile_name, raw_roles in cfg.control.aim.class_roles.items():
        if profile_name not in cfg.inference.detection_class_profiles:
            raise ValueError(
                "runtime config key 'control.aim.class_roles' references unknown "
                f"detection profile: {profile_name}"
            )
        if not isinstance(raw_roles, dict):
            raise ValueError(
                f"runtime config key 'control.aim.class_roles.{profile_name}' must be a mapping"
            )
        roles: dict[str, str] = {}
        for raw_class_id, raw_role in raw_roles.items():
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "runtime config class role keys must be class indexes"
                ) from exc
            if class_id < 0 or class_id > 255:
                raise ValueError("runtime config class role indexes must be in [0, 255]")
            role = str(raw_role).strip().lower()
            if role not in {"head", "body", "other"}:
                raise ValueError("runtime config class roles must be head, body, or other")
            roles[str(class_id)] = role
        normalized_class_roles[str(profile_name)] = roles
    cfg.control.aim.class_roles = normalized_class_roles
    bounded_controls = {
        "configured_actuation_delay_s": (0.0, 0.1),
        "scheduler_interval_ms": (1.0, 10.0),
        "target_switch_delay_ms": (0.0, 500.0),
    }
    for key, (minimum, maximum) in bounded_controls.items():
        value = float(getattr(cfg.control, key))
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(
                f"runtime config key 'control.{key}' must be >= {minimum} and <= {maximum}"
            )
    for key in ("target_fov_radius_px",):
        if (
            not math.isfinite(float(getattr(cfg.control, key)))
            or float(getattr(cfg.control, key)) <= 0.0
        ):
            raise ValueError(f"runtime config key 'control.{key}' must be finite and > 0")
    calibrated = cfg.control.calibrated_angular
    calibrated_bounds = {
        "fov_x_deg": (30.0, 179.0),
        "kp_x": (0.0, 2.0),
        "kp_y": (0.0, 2.0),
        "kd_x": (0.0, 1.0),
        "kd_y": (0.0, 1.0),
        "d_ema_alpha": (0.01, 1.0),
    }
    for key, (minimum, maximum) in calibrated_bounds.items():
        value = float(getattr(calibrated, key))
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(
                f"runtime config key 'control.calibrated_angular.{key}' must be >= {minimum} and <= {maximum}"
            )
    for key in (
        "counts_per_360_x",
        "counts_per_360_y",
        "max_angle_step_x_deg",
        "max_angle_step_y_deg",
    ):
        value = float(getattr(calibrated, key))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"runtime config key 'control.calibrated_angular.{key}' must be finite and > 0"
            )
    universal = cfg.control.universal_saturated
    for key in (
        "response_scale_x_px",
        "response_scale_y_px",
        "max_step_x_counts",
        "max_step_y_counts",
    ):
        value = float(getattr(universal, key))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"runtime config key 'control.universal_saturated.{key}' must be finite and > 0"
            )
    _validate_dual_phase_robust_v2_algorithm(cfg.control.dual_phase_atan_robust_predictive_v2)
    shared = cfg.control.shared
    for key in ("deadzone_x_px", "deadzone_y_px"):
        value = float(getattr(shared, key))
        if not math.isfinite(value) or value < 0.0 or value > 10.0:
            raise ValueError(f"runtime config key 'control.shared.{key}' must be >= 0 and <= 10")
    for key in ("max_count_slew_x", "max_count_slew_y"):
        value = float(getattr(shared, key))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"runtime config key 'control.shared.{key}' must be finite and > 0")
    shared_bounds = {
        "trigger_activation_delay_ms": (0.0, 1000.0),
    }
    for key, (minimum, maximum) in shared_bounds.items():
        value = float(getattr(shared, key))
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(
                f"runtime config key 'control.shared.{key}' must be >= {minimum} and <= {maximum}"
            )
    recoil = cfg.control.recoil
    recoil_bounds = {
        "base_rate_counts_s": (0.0, 20000.0),
        "max_rate_counts_s": (0.0, 20000.0),
        "startup_ms": (0.0, 1000.0),
        "positive_deadzone_norm": (0.0, 1.0),
        "negative_deadzone_norm": (0.0, 1.0),
        "full_brake_error_norm": (0.0, 1.0),
        "fast_add_gain_counts_s": (0.0, 20000.0),
        "max_fast_add_ratio": (0.0, 1.0),
        "stale_threshold_ms": (0.0, 5000.0),
    }
    for key, (minimum, maximum) in recoil_bounds.items():
        value = float(getattr(recoil, key))
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(f"runtime config key 'control.recoil.{key}' must be finite and within bounds")
    if recoil.max_rate_counts_s < recoil.base_rate_counts_s:
        raise ValueError("runtime config key 'control.recoil.max_rate_counts_s' must be >= base_rate_counts_s")
    if recoil.full_brake_error_norm <= recoil.negative_deadzone_norm:
        raise ValueError("runtime config key 'control.recoil.full_brake_error_norm' must be > negative_deadzone_norm")
    for key in ("scheduler_step_counts_x", "scheduler_step_counts_y"):
        value = int(getattr(cfg.control, key))
        if value < 1 or value > 20:
            raise ValueError(f"runtime config key 'control.{key}' must be >= 1 and <= 20")
    if cfg.control.trigger_mode not in {"hardware", "always"}:
        raise ValueError("runtime config key 'control.trigger_mode' must be hardware or always")
    if cfg.control.candidate_ratio_max_aspect < 1:
        raise ValueError("runtime config key 'control.candidate_ratio_max_aspect' must be >= 1")
    selection_weight_keys = (
        "candidate_selection_class_weight",
        "candidate_selection_distance_weight",
    )
    for key in selection_weight_keys:
        value = float(getattr(cfg.control, key))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"runtime config key 'control.{key}' must be finite and >= 0")
    if sum(float(getattr(cfg.control, key)) for key in selection_weight_keys) <= 0.0:
        raise ValueError("runtime config target selection weights must have a positive sum")
    if cfg.control.tracker_max_match_distance <= 0:
        raise ValueError("runtime config key 'control.tracker_max_match_distance' must be > 0")
    if cfg.control.tracker_position_cost_weight < 0:
        raise ValueError("runtime config key 'control.tracker_position_cost_weight' must be >= 0")
    if cfg.control.tracker_iou_cost_weight < 0:
        raise ValueError("runtime config key 'control.tracker_iou_cost_weight' must be >= 0")
    if cfg.control.tracker_position_cost_weight + cfg.control.tracker_iou_cost_weight <= 0:
        raise ValueError(
            "runtime config key 'control.tracker_position_cost_weight' and 'control.tracker_iou_cost_weight' must have a positive sum"
        )
    if cfg.control.tracker_max_missed_frames < 0:
        raise ValueError("runtime config key 'control.tracker_max_missed_frames' must be >= 0")
    if (
        cfg.control.target_switch_min_preference_advantage < 0
        or cfg.control.target_switch_min_preference_advantage > 1
    ):
        raise ValueError(
            "runtime config key 'control.target_switch_min_preference_advantage' must be >= 0 and <= 1"
        )
    if (
        cfg.control.target_switch_min_continuity_score < 0
        or cfg.control.target_switch_min_continuity_score > 1
    ):
        raise ValueError(
            "runtime config key 'control.target_switch_min_continuity_score' must be >= 0 and <= 1"
        )
    for key in (
        "kalman_acceleration_noise",
        "kalman_measurement_noise_x",
        "kalman_measurement_noise_y",
        "kalman_max_predict_missing_ms",
        "kalman_max_predict_dt_ms",
        "kalman_max_position_sigma_px",
        "kalman_max_covariance_trace",
        "kalman_nis_threshold",
        "kalman_nis_hard_reject",
        "kalman_prediction_decay_tau_ms",
    ):
        if getattr(cfg.control, key) <= 0:
            raise ValueError(f"runtime config key 'control.{key}' must be > 0")
    if cfg.control.kalman_max_predict_steps < 1:
        raise ValueError("runtime config key 'control.kalman_max_predict_steps' must be >= 1")
    for key in (
        "kalman_min_identity_confidence",
        "kalman_min_prediction_confidence",
    ):
        value = getattr(cfg.control, key)
        if value < 0 or value > 1:
            raise ValueError(f"runtime config key 'control.{key}' must be >= 0 and <= 1")
    if cfg.control.kalman_nis_hard_reject < cfg.control.kalman_nis_threshold:
        raise ValueError(
            "runtime config key 'control.kalman_nis_hard_reject' must be >= control.kalman_nis_threshold"
        )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return RuntimeConfig()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {cfg_path}")
    raw = _drop_legacy_runtime_keys(raw)
    cfg = _build_dataclass(RuntimeConfig, raw)
    _validate_runtime_rules(cfg)
    return cfg


def parse_runtime_config(raw: dict[str, Any]) -> RuntimeConfig:
    if not isinstance(raw, dict):
        raise ValueError("runtime config must be a mapping")
    raw = _drop_legacy_runtime_keys(raw)
    cfg = _build_dataclass(RuntimeConfig, raw)
    _validate_runtime_rules(cfg)
    return cfg


def _parse_class_priority(value: str) -> list[int]:
    if not value.strip():
        return []
    result: list[int] = []
    seen: set[int] = set()
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            class_id = int(text)
        except ValueError as exc:
            raise ValueError(
                "runtime config key 'inference.detection_class_priority' must be comma-separated class indexes"
            ) from exc
        if class_id < 0 or class_id > 255:
            raise ValueError(
                "runtime config key 'inference.detection_class_priority' must contain class indexes between 0 and 255"
            )
        if class_id not in seen:
            seen.add(class_id)
            result.append(class_id)
    return result


def _parse_detection_class_filter(value: str) -> set[int]:
    if not value.strip():
        raise ValueError(
            "runtime config key 'inference.detection_class_filter' must be all or comma-separated class indexes"
        )
    result: set[int] = set()
    for part in value.split(","):
        text = part.strip()
        try:
            class_id = int(text)
        except ValueError as exc:
            raise ValueError(
                "runtime config key 'inference.detection_class_filter' must be all or comma-separated class indexes"
            ) from exc
        if class_id < 0 or class_id > 255:
            raise ValueError(
                "runtime config key 'inference.detection_class_filter' must contain class indexes between 0 and 255"
            )
        result.add(class_id)
    return result


def save_runtime_config(cfg: RuntimeConfig, path: str | Path) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
