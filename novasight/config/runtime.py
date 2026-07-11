from __future__ import annotations

from dataclasses import MISSING, Field, asdict, dataclass, field, fields, is_dataclass
import math
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
    game_sensitivity_fingerprint: str = "unverified-default"


@dataclass
class AimConfig:
    y_ratio: float = 0.22


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
class SharedControlConfig:
    max_count_slew_x: float = 10.0
    max_count_slew_y: float = 8.0
    deadzone_x_px: float = 0.0
    deadzone_y_px: float = 0.0
    invert_y: bool = False


@dataclass
class ControlConfig:
    mode: str = "universal_saturated"
    min_confidence: float = 0.25
    target_fov_radius_px: float = 180.0
    target_switch_delay_ms: float = 50.0
    target_lock_enabled: bool = True
    target_sticky_bias: float = 0.25
    candidate_ratio_max_aspect: float = 6.0
    candidate_quality_confidence_weight: float = 0.7
    candidate_quality_area_weight: float = 0.3
    class_priority_quality_margin: float = 0.08
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
    calibrated_angular: CalibratedAngularConfig = field(default_factory=CalibratedAngularConfig)
    universal_saturated: UniversalSaturatedConfig = field(default_factory=UniversalSaturatedConfig)
    shared: SharedControlConfig = field(default_factory=SharedControlConfig)
    configured_actuation_delay_s: float = 0.004
    prediction_strength: float = 1.0
    prediction_x_enabled: bool = True
    prediction_y_enabled: bool = True
    scheduler_enabled: bool = True
    scheduler_step_counts_x: int = 32
    scheduler_step_counts_y: int = 32
    scheduler_interval_ms: float = 4.0
    trigger_mode: str = "always"


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
    min_effective_move_counts_x: int = 16
    min_effective_move_counts_y: int = 16


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
    hardware = normalized.get("hardware")
    if isinstance(hardware, dict):
        hardware = dict(hardware)
        # The preceding kmNet executor accepted this key but forced it to False.
        hardware.pop("flip_dy", None)
        for key in ("kind", "serial_port", "heartbeat_timeout_ms"):
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
        raise ValueError(
            "legacy config key 'control.strategy' must be experimental_angle_pid"
        )
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
        delay_ms = (
            legacy_delay_ms
            if legacy_delay_ms is not None
            else legacy_estimated_delay_ms
        )
        if delay_ms is not None:
            control["configured_actuation_delay_s"] = (
                _legacy_number(delay_ms, "control.configured_extra_prediction_delay_ms")
                / 1000.0
            )

    legacy_prediction_enabled = control.pop("latency_compensation_enabled", None)
    if legacy_prediction_enabled is False:
        control.setdefault("prediction_x_enabled", False)
        control.setdefault("prediction_y_enabled", False)
    _move_legacy_number(
        control,
        "latency_compensation_scale",
        "prediction_strength",
    )
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

    min_confidence = control.get("min_confidence")
    if (
        legacy_schema
        and isinstance(min_confidence, (int, float))
        and not isinstance(min_confidence, bool)
    ):
        control["min_confidence"] = max(0.10, float(min_confidence))

    for key in tuple(control):
        if key.startswith("experimental_angle_"):
            control.pop(key, None)
    for key in _REMOVED_LEGACY_CONTROL_KEYS:
        control.pop(key, None)


def _migrate_dual_control_modes(
    control: dict[str, Any],
    calibration: dict[str, Any],
) -> None:
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

    legacy_calibrated_fields = {
        "kp_x": "kp_x",
        "kp_y": "kp_y",
        "kd_x": "kd_x",
        "kd_y": "kd_y",
        "d_ema_alpha": "d_ema_alpha",
    }
    legacy_mode_detected = any(key in control for key in legacy_calibrated_fields) or any(
        key in calibration
        for key in ("fov_x_deg", "counts_per_360_x", "counts_per_360_y")
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
    if not cfg.calibration.game_sensitivity_fingerprint.strip():
        raise ValueError("runtime config key 'calibration.game_sensitivity_fingerprint' must be non-empty")
    if cfg.control.mode not in {"calibrated_angular", "universal_saturated"}:
        raise ValueError(
            "runtime config key 'control.mode' must be calibrated_angular or universal_saturated"
        )
    if not math.isfinite(float(cfg.control.aim.y_ratio)):
        raise ValueError("runtime config key 'control.aim.y_ratio' must be finite")
    cfg.control.aim.y_ratio = round(max(0.0, min(1.0, float(cfg.control.aim.y_ratio))), 2)
    bounded_controls = {
        "configured_actuation_delay_s": (0.0, 0.1),
        "prediction_strength": (0.0, 1.5),
        "scheduler_interval_ms": (1.0, 10.0),
        "min_confidence": (0.10, 0.99),
        "target_switch_delay_ms": (0.0, 500.0),
    }
    for key, (minimum, maximum) in bounded_controls.items():
        value = float(getattr(cfg.control, key))
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(
                f"runtime config key 'control.{key}' must be >= {minimum} and <= {maximum}"
            )
    for key in ("target_fov_radius_px",):
        if not math.isfinite(float(getattr(cfg.control, key))) or float(getattr(cfg.control, key)) <= 0.0:
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
    shared = cfg.control.shared
    for key in ("deadzone_x_px", "deadzone_y_px"):
        value = float(getattr(shared, key))
        if not math.isfinite(value) or value < 0.0 or value > 10.0:
            raise ValueError(
                f"runtime config key 'control.shared.{key}' must be >= 0 and <= 10"
            )
    for key in ("max_count_slew_x", "max_count_slew_y"):
        value = float(getattr(shared, key))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"runtime config key 'control.shared.{key}' must be finite and > 0"
            )
    for key in ("scheduler_step_counts_x", "scheduler_step_counts_y"):
        value = int(getattr(cfg.control, key))
        if value < 1 or value > 64:
            raise ValueError(f"runtime config key 'control.{key}' must be >= 1 and <= 64")
    for key in ("min_effective_move_counts_x", "min_effective_move_counts_y"):
        value = int(getattr(cfg.hardware, key))
        if value < 1 or value > 64:
            raise ValueError(f"runtime config key 'hardware.{key}' must be >= 1 and <= 64")
    if cfg.control.trigger_mode not in {"hardware", "always"}:
        raise ValueError("runtime config key 'control.trigger_mode' must be hardware or always")
    if cfg.control.candidate_ratio_max_aspect < 1:
        raise ValueError("runtime config key 'control.candidate_ratio_max_aspect' must be >= 1")
    if cfg.control.class_priority_quality_margin > 1:
        raise ValueError("runtime config key 'control.class_priority_quality_margin' must be <= 1")
    if cfg.control.tracker_max_match_distance <= 0:
        raise ValueError("runtime config key 'control.tracker_max_match_distance' must be > 0")
    if cfg.control.tracker_position_cost_weight < 0:
        raise ValueError("runtime config key 'control.tracker_position_cost_weight' must be >= 0")
    if cfg.control.tracker_iou_cost_weight < 0:
        raise ValueError("runtime config key 'control.tracker_iou_cost_weight' must be >= 0")
    if cfg.control.tracker_position_cost_weight + cfg.control.tracker_iou_cost_weight <= 0:
        raise ValueError("runtime config key 'control.tracker_position_cost_weight' and 'control.tracker_iou_cost_weight' must have a positive sum")
    if cfg.control.tracker_max_missed_frames < 0:
        raise ValueError("runtime config key 'control.tracker_max_missed_frames' must be >= 0")
    if cfg.control.target_switch_min_preference_advantage < 0 or cfg.control.target_switch_min_preference_advantage > 1:
        raise ValueError("runtime config key 'control.target_switch_min_preference_advantage' must be >= 0 and <= 1")
    if cfg.control.target_switch_min_continuity_score < 0 or cfg.control.target_switch_min_continuity_score > 1:
        raise ValueError("runtime config key 'control.target_switch_min_continuity_score' must be >= 0 and <= 1")
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
        raise ValueError("runtime config key 'control.kalman_nis_hard_reject' must be >= control.kalman_nis_threshold")

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
