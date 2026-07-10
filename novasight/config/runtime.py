from __future__ import annotations

from dataclasses import MISSING, Field, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

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
class RoiConfig:
    size: int = 640
    mode: str = "center"
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class InferenceConfig:
    enabled: bool = True
    backend: str = "tensorrt"
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
    detection_class_profiles: dict[str, list[str]] = field(default_factory=lambda: {
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
    })


@dataclass
class PreprocessConfig:
    backend: str = "cpu"
    input_format: str = "auto"
    output_dtype: str = "fp16"
    normalize: bool = True
    use_pinned_memory: bool = True
    h2d_async: bool = True


@dataclass
class CaptureConfig:
    device: str = "/dev/video0"
    backend: str = "gst_cpu_latest"
    preference: str = "auto_high_fps"
    memory: str = "system"
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
    fov_semantics: str = "horizontal"
    fov_x_deg: float = 105.0
    counts_per_360_x: float = 9980.0
    counts_per_360_y: float = 9980.0
    axis_sign_x: float = 1.0
    axis_sign_y: float = 1.0
    game_sensitivity_fingerprint: str = "unverified-default"
    projection_profile: str = "fixed_horizontal_fov"


@dataclass
class ControlConfig:
    max_abs_dx: int = 200
    max_abs_dy: int = 200
    min_confidence: float = 0.0
    fov_ratio: float = 0.28
    target_lock_enabled: bool = True
    target_sticky_bias: float = 0.25
    target_lost_grace_frames: int = 5
    candidate_ratio_max_aspect: float = 6.0
    candidate_quality_confidence_weight: float = 0.7
    candidate_quality_area_weight: float = 0.3
    class_priority_quality_margin: float = 0.08
    tracker_confirm_frames: int = 2
    target_switch_min_preference_advantage: float = 0.08
    target_switch_min_continuity_score: float = 0.70
    target_switch_confirm_frames: int = 3
    tracker_matching_distance_px: float = 140.0
    tracker_ambiguity_margin: float = 0.08
    tracker_missing_timeout_ms: float = 120.0
    tracker_delete_timeout_ms: float = 250.0
    tracker_match_threshold: float = 0.65
    tracker_mahalanobis_gate: float = 9.21
    kalman_enabled: bool = True
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
    aim_horizontal_percent: float = 50.0
    aim_ratio: float = 40.0
    aim_offset_x_px: float = 0.0
    aim_offset_y_px: float = 0.0
    aim_ema_enabled: bool = True
    aim_ema_alpha: float = 0.65
    aim_max_anchor_jump_ratio: float = 0.15
    latency_compensation_enabled: bool = True
    latency_compensation_scale: float = 0.70
    latency_max_compensation_ms: float = 35.0
    latency_reject_if_age_exceeds_ms: float = 55.0
    latency_max_compensation_px: float = 80.0
    latency_min_velocity_px_s: float = 30.0
    latency_max_velocity_px_s: float = 2500.0
    latency_min_velocity_measurements: int = 3
    latency_min_velocity_confidence: float = 0.65
    configured_extra_prediction_delay_ms: float = 2.0
    trigger_mode: str = "hardware"
    output_mode: str = "kmnet"
    strategy: str = "experimental_angle_pid"
    command_interval_ms: float = 1.0
    scheduler_command_ttl_ms: float = 35.0
    scheduler_predicted_command_ttl_ms: float = 18.0
    scheduler_cancel_on_new_frame: bool = True
    scheduler_cancel_on_direction_change: bool = True
    scheduler_cancel_on_track_change: bool = True
    scheduler_max_step_x: int = 20
    scheduler_max_step_y: int = 20
    scheduler_queue_hard_limit: int = 64
    scheduler_device_error_cooldown_ms: float = 50.0
    move_kind: str = "bezier"
    move_ms: int = 12
    trace_ms: int = 0
    bezier_curvature: float = 0.18
    experimental_angle_kp_x: float = 0.35
    experimental_angle_kp_y: float = 0.24
    experimental_angle_ki: float = 0.0
    experimental_angle_kd: float = 0.0
    experimental_angle_integral_limit: float = 0.0
    experimental_angle_config_level: str = "basic"
    experimental_angle_speed: float = 1.0
    experimental_angle_smooth_factor: float = 0.0
    experimental_angle_deadzone_px: float = 0.0
    experimental_angle_derivative_filter: float = 1.0
    experimental_angle_near_error_deg: float = 0.35
    experimental_angle_far_error_deg: float = 2.50
    experimental_angle_near_kp_scale: float = 0.35
    experimental_angle_middle_kp_scale: float = 0.70
    experimental_angle_far_kp_scale: float = 1.00
    experimental_angle_near_kd_scale: float = 1.00
    experimental_angle_middle_kd_scale: float = 0.80
    experimental_angle_far_kd_scale: float = 0.50
    experimental_angle_prediction_gain_min: float = 0.35
    experimental_angle_prediction_d_gain_min: float = 0.25
    experimental_angle_max_control_angle_deg: float = 3.0
    experimental_angle_max_step_counts: float = 80.0
    experimental_angle_max_counts_delta_x: float = 35.0
    experimental_angle_max_counts_delta_y: float = 35.0
    experimental_angle_control_hz: float = 60.0
    experimental_angle_kalman_enabled: bool = True
    experimental_angle_kalman_process_noise: float = 2.0
    experimental_angle_kalman_measurement_noise: float = 16.0
    experimental_angle_hungarian_enabled: bool = True
    experimental_angle_matching_distance_px: float = 140.0
    experimental_angle_max_extrapolate_frames: int = 3
    experimental_angle_target_filter_enabled: bool = True
    experimental_angle_target_filter_min_score: float = 0.0
    experimental_angle_target_filter_fov_ratio: float = 1.0
    experimental_angle_target_filter_same_class: bool = False
    experimental_angle_prediction_lead_ms: float = 0.0
    experimental_angle_extrapolate_confidence_decay: float = 1.0
    experimental_angle_magnet_enabled: bool = False
    experimental_angle_magnet_radius_px: float = 120.0
    experimental_angle_magnet_strength: float = 0.25
    experimental_angle_magnet_curve: float = 1.0
    experimental_angle_magnet_deadzone_px: float = 0.0
    experimental_angle_magnet_max_counts: float = 20.0


@dataclass
class ExecutorConfig:
    default: str = "kmnet"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "logs"


@dataclass
class HardwareConfig:
    kind: str = "kmnet"
    host: str = "192.168.2.188"
    port: int = 8888
    uuid: str = "12345678"
    monitor_port: int = 5001
    flip_dy: bool = False
    serial_port: str = ""
    heartbeat_timeout_ms: float = 50.0


@dataclass
class RuntimeConfig:
    web: WebConfig = field(default_factory=WebConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    consumers: ConsumerConfig = field(default_factory=ConsumerConfig)
    limits: RuntimeLimitsConfig = field(default_factory=RuntimeLimitsConfig)
    runtime: RuntimeBehaviorConfig = field(default_factory=RuntimeBehaviorConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
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


def _build_dataclass(cls: type[T], raw: dict[str, Any], section: str = "") -> T:
    items = {item.name: item for item in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown_keys = sorted(set(raw) - set(items), key=str)
    if unknown_keys:
        key_names = ", ".join(
            f"{section}.{key}" if section else str(key) for key in unknown_keys
        )
        raise ValueError(f"unknown config key(s): {key_names}")

    values: dict[str, Any] = {}
    for item in items.values():
        if item.name not in raw:
            continue
        value = raw[item.name]
        current = _field_default(item)
        if is_dataclass(current) and isinstance(value, dict):
            key_name = f"{section}.{item.name}" if section else item.name
            values[item.name] = _build_dataclass(type(current), value, key_name)
        elif is_dataclass(current):
            key_name = f"{section}.{item.name}" if section else item.name
            raise ValueError(f"runtime config section '{key_name}' must be a mapping")
        else:
            key_name = f"{section}.{item.name}" if section else item.name
            _validate_leaf_value(key_name, value, type_hints[item.name])
            values[item.name] = value
    return cls(**values)


def _drop_legacy_runtime_keys(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    inference = normalized.get("inference")
    if isinstance(inference, dict):
        inference = dict(inference)
        if str(inference.get("backend", "")).lower() in {
            "deepstream",
            "onnxruntime",
            "legacy_latest",
            "deepstream_uncontrolled",
        }:
            inference["backend"] = "tensorrt"
        for key in (
            "deepstream_manifest_path",
            "deepstream_config_path",
            "deepstream_io_mode",
            "deepstream_batched_push_timeout_us",
            "deepstream_tracker_config_path",
        ):
            inference.pop(key, None)
        normalized["inference"] = inference
    capture = normalized.get("capture")
    if isinstance(capture, dict):
        capture = dict(capture)
        memory = str(capture.get("memory", "")).lower()
        if memory == "cpu":
            capture["memory"] = "system"
        if str(capture.get("backend", "")).lower() in {
            "deepstream",
            "legacy_latest",
            "deepstream_uncontrolled",
        }:
            capture["backend"] = "gst_cpu_latest"
        normalized["capture"] = capture
    control = normalized.get("control")
    if isinstance(control, dict):
        control = dict(control)
        legacy_delay = control.pop("latency_estimated_actuation_delay_ms", None)
        if (
            legacy_delay is not None
            and "configured_extra_prediction_delay_ms" not in control
        ):
            control["configured_extra_prediction_delay_ms"] = legacy_delay
        calibration = normalized.get("calibration")
        if calibration is None:
            calibration = {}
        if isinstance(calibration, dict):
            calibration = dict(calibration)
            if (
                "experimental_angle_fov_x_deg" in control
                and "fov_x_deg" not in calibration
            ):
                calibration["fov_x_deg"] = control["experimental_angle_fov_x_deg"]
            if "experimental_angle_counts_per_360" in control:
                if "counts_per_360_x" not in calibration:
                    calibration["counts_per_360_x"] = control[
                        "experimental_angle_counts_per_360"
                    ]
                if "counts_per_360_y" not in calibration:
                    calibration["counts_per_360_y"] = control[
                        "experimental_angle_counts_per_360"
                    ]
            if "experimental_angle_sign_x" in control and "axis_sign_x" not in calibration:
                calibration["axis_sign_x"] = control["experimental_angle_sign_x"]
            if "experimental_angle_sign_y" in control and "axis_sign_y" not in calibration:
                calibration["axis_sign_y"] = control["experimental_angle_sign_y"]
            normalized["calibration"] = calibration
        for key in (
            "experimental_angle_fov_x_deg",
            "experimental_angle_counts_per_360",
            "experimental_angle_sign_x",
            "experimental_angle_sign_y",
        ):
            control.pop(key, None)
        normalized["control"] = control
    return normalized


def _validate_runtime_rules(cfg: RuntimeConfig) -> None:
    if cfg.source.default not in {"null", "capture", "image"} and not cfg.source.default.startswith("image:"):
        raise ValueError("runtime config key 'source.default' must be one of null, capture, image, or image:<path>")
    if cfg.source.image_fps not in {1, 5, 15, 30, 60}:
        raise ValueError("runtime config key 'source.image_fps' must be one of 1, 5, 15, 30, 60")
    if cfg.capture.backend not in {"gst_cpu_latest", "nvmm_latest"}:
        raise ValueError("runtime config key 'capture.backend' must be gst_cpu_latest or nvmm_latest")
    if cfg.capture.memory not in {"system", "nvmm"}:
        raise ValueError("runtime config key 'capture.memory' must be system or nvmm")
    if cfg.capture.backend == "gst_cpu_latest" and cfg.capture.memory != "system":
        raise ValueError("runtime config key 'capture.memory' must be system for gst_cpu_latest")
    if cfg.capture.backend == "nvmm_latest" and cfg.capture.memory != "nvmm":
        raise ValueError("runtime config key 'capture.memory' must be nvmm for nvmm_latest")
    if not cfg.capture.latest_only:
        raise ValueError("runtime config key 'capture.latest_only' must be true")
    if cfg.capture.appsink_max_buffers != 1:
        raise ValueError("runtime config key 'capture.appsink_max_buffers' must be 1")
    if cfg.capture.queue_leaky != "downstream":
        raise ValueError("runtime config key 'capture.queue_leaky' must be downstream")
    if cfg.limits.stream_fps not in {15, 30, 60}:
        raise ValueError("runtime config key 'limits.stream_fps' must be one of 15, 30, 60")
    if cfg.runtime.freshness_threshold_ms < 0:
        raise ValueError("runtime config key 'runtime.freshness_threshold_ms' must be >= 0")
    if not cfg.runtime.drop_stale_batches:
        raise ValueError("runtime config key 'runtime.drop_stale_batches' must be true")
    if not cfg.runtime.consume_latest_only:
        raise ValueError("runtime config key 'runtime.consume_latest_only' must be true")
    if cfg.consumers.recording_format not in {"csv", "parquet"}:
        raise ValueError("runtime config key 'consumers.recording_format' must be csv or parquet")
    try:
        cfg.roi.size = normalize_roi_size(cfg.roi.size)
    except ValueError as exc:
        allowed = ", ".join(str(size) for size in ROI_SIZE_CHOICES)
        raise ValueError(f"unsupported ROI size: {cfg.roi.size}; must be one of {allowed}") from exc
    if cfg.roi.mode not in {"center", "manual"}:
        raise ValueError("unsupported ROI mode: must be center or manual")
    if cfg.preprocess.backend not in {"cpu", "cuda"}:
        raise ValueError("runtime config key 'preprocess.backend' must be cpu or cuda")
    if cfg.preprocess.input_format != "auto":
        raise ValueError("runtime config key 'preprocess.input_format' must be auto")
    if cfg.preprocess.output_dtype not in {"fp16", "fp32", "float16", "float32"}:
        raise ValueError("runtime config key 'preprocess.output_dtype' must be fp16 or fp32")
    if cfg.inference.backend not in {"tensorrt", "nvmm_latest"}:
        raise ValueError(
            "runtime config key 'inference.backend' must be tensorrt or nvmm_latest"
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
        raise ValueError("runtime config key 'inference.confidence_threshold' must be >= 0 and <= 1")
    if cfg.inference.nms_threshold < 0 or cfg.inference.nms_threshold > 1:
        raise ValueError("runtime config key 'inference.nms_threshold' must be >= 0 and <= 1")
    if cfg.inference.input_source != "source.default":
        raise ValueError("runtime config key 'inference.input_source' must be source.default")
    if cfg.inference.detection_class_filter != "all":
        try:
            class_index = int(cfg.inference.detection_class_filter)
        except ValueError as exc:
            raise ValueError("runtime config key 'inference.detection_class_filter' must be all or a class index") from exc
        if class_index < 0 or class_index > 255:
            raise ValueError("runtime config key 'inference.detection_class_filter' must be between 0 and 255")
    _parse_class_priority(cfg.inference.detection_class_priority)
    if cfg.inference.detection_class_profile not in cfg.inference.detection_class_profiles:
        raise ValueError("runtime config key 'inference.detection_class_profile' must exist in detection_class_profiles")
    if not cfg.calibration.profile_id.strip():
        raise ValueError("runtime config key 'calibration.profile_id' must be non-empty")
    if cfg.calibration.profile_version < 1:
        raise ValueError("runtime config key 'calibration.profile_version' must be >= 1")
    if cfg.calibration.fov_semantics != "horizontal":
        raise ValueError("runtime config key 'calibration.fov_semantics' must be horizontal")
    if cfg.calibration.fov_x_deg <= 0 or cfg.calibration.fov_x_deg >= 180:
        raise ValueError("runtime config key 'calibration.fov_x_deg' must be > 0 and < 180")
    if cfg.calibration.counts_per_360_x < 1:
        raise ValueError("runtime config key 'calibration.counts_per_360_x' must be >= 1")
    if cfg.calibration.counts_per_360_y < 1:
        raise ValueError("runtime config key 'calibration.counts_per_360_y' must be >= 1")
    if cfg.calibration.axis_sign_x not in {-1, 1, -1.0, 1.0}:
        raise ValueError("runtime config key 'calibration.axis_sign_x' must be -1 or 1")
    if cfg.calibration.axis_sign_y not in {-1, 1, -1.0, 1.0}:
        raise ValueError("runtime config key 'calibration.axis_sign_y' must be -1 or 1")
    if not cfg.calibration.game_sensitivity_fingerprint.strip():
        raise ValueError("runtime config key 'calibration.game_sensitivity_fingerprint' must be non-empty")
    if cfg.calibration.projection_profile != "fixed_horizontal_fov":
        raise ValueError("runtime config key 'calibration.projection_profile' must be fixed_horizontal_fov")
    if cfg.control.strategy != "experimental_angle_pid":
        raise ValueError("runtime config key 'control.strategy' must be experimental_angle_pid")
    if cfg.control.fov_ratio <= 0 or cfg.control.fov_ratio > 1:
        raise ValueError("runtime config key 'control.fov_ratio' must be > 0 and <= 1")
    if cfg.control.target_sticky_bias < 0 or cfg.control.target_sticky_bias > 0.9:
        raise ValueError("runtime config key 'control.target_sticky_bias' must be >= 0 and <= 0.9")
    if cfg.control.aim_ratio < 0 or cfg.control.aim_ratio > 100:
        raise ValueError("runtime config key 'control.aim_ratio' must be >= 0 and <= 100")
    if cfg.control.aim_horizontal_percent < 0 or cfg.control.aim_horizontal_percent > 100:
        raise ValueError("runtime config key 'control.aim_horizontal_percent' must be >= 0 and <= 100")
    for key in ("aim_offset_x_px", "aim_offset_y_px"):
        value = getattr(cfg.control, key)
        if value < -200 or value > 200:
            raise ValueError(f"runtime config key 'control.{key}' must be >= -200 and <= 200")
    if cfg.control.aim_ema_alpha < 0 or cfg.control.aim_ema_alpha > 1:
        raise ValueError("runtime config key 'control.aim_ema_alpha' must be >= 0 and <= 1")
    if cfg.control.aim_max_anchor_jump_ratio < 0 or cfg.control.aim_max_anchor_jump_ratio > 1:
        raise ValueError("runtime config key 'control.aim_max_anchor_jump_ratio' must be >= 0 and <= 1")
    if cfg.control.latency_compensation_scale < 0 or cfg.control.latency_compensation_scale > 1.5:
        raise ValueError("runtime config key 'control.latency_compensation_scale' must be >= 0 and <= 1.5")
    for key in (
        "latency_max_compensation_ms",
        "latency_reject_if_age_exceeds_ms",
        "latency_max_compensation_px",
        "latency_max_velocity_px_s",
    ):
        if getattr(cfg.control, key) <= 0:
            raise ValueError(f"runtime config key 'control.{key}' must be > 0")
    if cfg.control.latency_min_velocity_px_s < 0:
        raise ValueError("runtime config key 'control.latency_min_velocity_px_s' must be >= 0")
    if cfg.control.configured_extra_prediction_delay_ms < 0:
        raise ValueError(
            "runtime config key 'control.configured_extra_prediction_delay_ms' must be >= 0"
        )
    if cfg.control.latency_min_velocity_measurements < 1:
        raise ValueError("runtime config key 'control.latency_min_velocity_measurements' must be >= 1")
    if cfg.control.latency_min_velocity_confidence < 0 or cfg.control.latency_min_velocity_confidence > 1:
        raise ValueError("runtime config key 'control.latency_min_velocity_confidence' must be >= 0 and <= 1")
    if cfg.control.latency_max_velocity_px_s < cfg.control.latency_min_velocity_px_s:
        raise ValueError("runtime config key 'control.latency_max_velocity_px_s' must be >= control.latency_min_velocity_px_s")
    if cfg.control.trigger_mode not in {"hardware", "always"}:
        raise ValueError("runtime config key 'control.trigger_mode' must be hardware or always")
    if cfg.control.output_mode not in {"", "kmnet"}:
        raise ValueError("runtime config key 'control.output_mode' must be kmnet")
    if cfg.executor.default != "kmnet":
        raise ValueError("runtime config key 'executor.default' must be kmnet")
    if cfg.hardware.kind != "kmnet":
        raise ValueError("runtime config key 'hardware.kind' must be kmnet")
    if cfg.control.target_lost_grace_frames < 0:
        raise ValueError("runtime config key 'control.target_lost_grace_frames' must be >= 0")
    if cfg.control.candidate_ratio_max_aspect < 1:
        raise ValueError("runtime config key 'control.candidate_ratio_max_aspect' must be >= 1")
    if cfg.control.class_priority_quality_margin > 1:
        raise ValueError("runtime config key 'control.class_priority_quality_margin' must be <= 1")
    if cfg.control.tracker_confirm_frames < 1:
        raise ValueError("runtime config key 'control.tracker_confirm_frames' must be >= 1")
    if cfg.control.target_switch_min_preference_advantage < 0 or cfg.control.target_switch_min_preference_advantage > 1:
        raise ValueError("runtime config key 'control.target_switch_min_preference_advantage' must be >= 0 and <= 1")
    if cfg.control.target_switch_min_continuity_score < 0 or cfg.control.target_switch_min_continuity_score > 1:
        raise ValueError("runtime config key 'control.target_switch_min_continuity_score' must be >= 0 and <= 1")
    if cfg.control.target_switch_confirm_frames < 1:
        raise ValueError("runtime config key 'control.target_switch_confirm_frames' must be >= 1")
    if cfg.control.tracker_match_threshold < 0 or cfg.control.tracker_match_threshold > 1:
        raise ValueError("runtime config key 'control.tracker_match_threshold' must be >= 0 and <= 1")
    for key in (
        "tracker_mahalanobis_gate",
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
        raise ValueError("runtime config key 'control.kalman_nis_hard_reject' must be >= control.kalman_nis_threshold")
    if cfg.control.move_kind not in {"raw", "enc_raw", "auto", "enc_auto", "bezier", "enc_bezier"}:
        raise ValueError("runtime config key 'control.move_kind' must be raw, enc_raw, auto, enc_auto, bezier, or enc_bezier")
    if cfg.control.scheduler_command_ttl_ms <= 0:
        raise ValueError("runtime config key 'control.scheduler_command_ttl_ms' must be > 0")
    if cfg.control.scheduler_predicted_command_ttl_ms <= 0:
        raise ValueError("runtime config key 'control.scheduler_predicted_command_ttl_ms' must be > 0")
    if cfg.control.scheduler_max_step_x <= 0:
        raise ValueError("runtime config key 'control.scheduler_max_step_x' must be > 0")
    if cfg.control.scheduler_max_step_y <= 0:
        raise ValueError("runtime config key 'control.scheduler_max_step_y' must be > 0")
    if cfg.control.scheduler_queue_hard_limit <= 0:
        raise ValueError("runtime config key 'control.scheduler_queue_hard_limit' must be > 0")
    for key in (
        "move_ms",
        "trace_ms",
        "command_interval_ms",
        "scheduler_command_ttl_ms",
        "scheduler_predicted_command_ttl_ms",
        "scheduler_max_step_x",
        "scheduler_max_step_y",
        "scheduler_queue_hard_limit",
        "scheduler_device_error_cooldown_ms",
        "candidate_quality_confidence_weight",
        "candidate_quality_area_weight",
        "class_priority_quality_margin",
        "target_switch_min_preference_advantage",
        "target_switch_min_continuity_score",
        "tracker_matching_distance_px",
        "tracker_ambiguity_margin",
        "tracker_missing_timeout_ms",
        "tracker_delete_timeout_ms",
        "tracker_mahalanobis_gate",
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
        "bezier_curvature",
        "experimental_angle_kp_x",
        "experimental_angle_kp_y",
        "experimental_angle_ki",
        "experimental_angle_integral_limit",
        "experimental_angle_speed",
        "experimental_angle_smooth_factor",
        "experimental_angle_deadzone_px",
        "experimental_angle_derivative_filter",
        "experimental_angle_near_error_deg",
        "experimental_angle_far_error_deg",
        "experimental_angle_near_kp_scale",
        "experimental_angle_middle_kp_scale",
        "experimental_angle_far_kp_scale",
        "experimental_angle_near_kd_scale",
        "experimental_angle_middle_kd_scale",
        "experimental_angle_far_kd_scale",
        "experimental_angle_prediction_gain_min",
        "experimental_angle_prediction_d_gain_min",
        "experimental_angle_max_control_angle_deg",
        "experimental_angle_max_step_counts",
        "experimental_angle_max_counts_delta_x",
        "experimental_angle_max_counts_delta_y",
        "experimental_angle_control_hz",
        "experimental_angle_kalman_process_noise",
        "experimental_angle_kalman_measurement_noise",
        "experimental_angle_matching_distance_px",
        "experimental_angle_target_filter_min_score",
        "experimental_angle_target_filter_fov_ratio",
        "experimental_angle_prediction_lead_ms",
        "experimental_angle_extrapolate_confidence_decay",
        "experimental_angle_magnet_radius_px",
        "experimental_angle_magnet_strength",
        "experimental_angle_magnet_curve",
        "experimental_angle_magnet_deadzone_px",
        "experimental_angle_magnet_max_counts",
    ):
        if getattr(cfg.control, key) < 0:
            raise ValueError(f"runtime config key 'control.{key}' must be >= 0")
    if cfg.control.experimental_angle_kd < -1 or cfg.control.experimental_angle_kd > 1:
        raise ValueError("runtime config key 'control.experimental_angle_kd' must be >= -1 and <= 1")
    if cfg.control.experimental_angle_config_level not in {"basic", "advanced", "developer"}:
        raise ValueError("runtime config key 'control.experimental_angle_config_level' must be basic, advanced, or developer")
    if cfg.control.experimental_angle_smooth_factor > 0.95:
        raise ValueError("runtime config key 'control.experimental_angle_smooth_factor' must be <= 0.95")
    if cfg.control.experimental_angle_derivative_filter > 1:
        raise ValueError("runtime config key 'control.experimental_angle_derivative_filter' must be <= 1")
    if cfg.control.experimental_angle_prediction_gain_min > 1:
        raise ValueError("runtime config key 'control.experimental_angle_prediction_gain_min' must be <= 1")
    if cfg.control.experimental_angle_prediction_d_gain_min > 1:
        raise ValueError("runtime config key 'control.experimental_angle_prediction_d_gain_min' must be <= 1")
    if cfg.control.experimental_angle_far_error_deg < cfg.control.experimental_angle_near_error_deg:
        raise ValueError("runtime config key 'control.experimental_angle_far_error_deg' must be >= control.experimental_angle_near_error_deg")
    if cfg.control.experimental_angle_max_control_angle_deg <= 0:
        raise ValueError("runtime config key 'control.experimental_angle_max_control_angle_deg' must be > 0")
    if cfg.control.experimental_angle_max_step_counts < 1:
        raise ValueError("runtime config key 'control.experimental_angle_max_step_counts' must be >= 1")
    if cfg.control.experimental_angle_max_counts_delta_x <= 0:
        raise ValueError("runtime config key 'control.experimental_angle_max_counts_delta_x' must be > 0")
    if cfg.control.experimental_angle_max_counts_delta_y <= 0:
        raise ValueError("runtime config key 'control.experimental_angle_max_counts_delta_y' must be > 0")
    if cfg.control.experimental_angle_control_hz < 1:
        raise ValueError("runtime config key 'control.experimental_angle_control_hz' must be >= 1")
    if cfg.control.experimental_angle_max_extrapolate_frames < 0:
        raise ValueError("runtime config key 'control.experimental_angle_max_extrapolate_frames' must be >= 0")
    if cfg.control.experimental_angle_target_filter_fov_ratio > 1:
        raise ValueError("runtime config key 'control.experimental_angle_target_filter_fov_ratio' must be <= 1")
    if cfg.control.experimental_angle_extrapolate_confidence_decay > 1:
        raise ValueError("runtime config key 'control.experimental_angle_extrapolate_confidence_decay' must be <= 1")

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


def save_runtime_config(cfg: RuntimeConfig, path: str | Path) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
