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


@dataclass
class RuntimeLimitsConfig:
    max_frame_queue: int = 1
    stream_fps: int = 30


@dataclass
class RoiConfig:
    size: int = 640
    mode: str = "center"
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class InferenceConfig:
    enabled: bool = True
    backend: str = "onnxruntime"
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
class CaptureConfig:
    device: str = "/dev/video0"
    preference: str = "auto_high_fps"
    pixel_format: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0


@dataclass
class ControlConfig:
    max_abs_dx: int = 200
    max_abs_dy: int = 200
    min_confidence: float = 0.0
    fov_ratio: float = 0.28
    target_lock_enabled: bool = True
    target_sticky_bias: float = 0.25
    target_lost_grace_frames: int = 5
    aim_ratio: float = 40.0
    trigger_mode: str = "hardware"
    trigger_bindings: list[str] = field(default_factory=lambda: ["MouseRight"])
    output_mode: str = ""
    strategy: str = "pid"
    pid_kp_x: float = 0.35
    pid_kp_y: float = 0.24
    pid_ki: float = 0.0
    pid_kd: float = 0.1
    pid_integral_limit: float = 250.0
    pid_move_limit: float = 120.0
    kp_x_move_max: float = 150.0
    kp_y_move_max: float = 30.0
    prediction_enabled: bool = True
    prediction_factor: float = 0.1
    derivative_enabled: bool = True
    y_down_enabled: bool = False
    command_interval_ms: float = 1.0
    y_rate_window_ms: float = 10.0
    y_rate_max_counts: float = 0.0
    move_kind: str = "bezier"
    move_ms: int = 12
    trace_ms: int = 0
    deadzone_counts: int = 1
    near_px: float = 24.0
    near_speed: float = 0.16
    far_speed: float = 0.42
    ema_alpha: float = 0.45
    counts_per_revolution_x: float = 9980.0
    counts_per_revolution_y: float = 9980.0
    bezier_curvature: float = 0.18
    straight_fov_deg: float = 105.0
    straight_c360: float = 9980.0
    straight_kp_x: float = 0.3
    straight_kp_y: float = 0.3
    straight_first_frame_gain: float = 1.0
    straight_first_max_step: float = 200.0
    straight_max_step: float = 80.0
    straight_in_deadzone: float = 8.0
    straight_jump_threshold: float = 40.0
    straight_pred_gain: float = 0.0
    straight_pred_consistency_frames: int = 3
    straight_gain_y: float = 1.0
    isolated_kp_x: float = 0.35
    isolated_kp_y: float = 0.24
    isolated_max_x: float = 80.0
    isolated_max_y: float = 60.0
    isolated_deadzone_px: float = 2.0
    isolated_aim_ratio: float = 40.0
    isolated_smoothing: float = 0.0
    isolated_prediction: float = 0.0
    isolated_fov_deg: float = 105.0
    isolated_counts_per_revolution_x: float = 9980.0
    isolated_counts_per_revolution_y: float = 9980.0
    dynamic_pid_kp_x: float = 0.35
    dynamic_pid_kp_y: float = 0.24
    dynamic_pid_ki: float = 0.0
    dynamic_pid_kd: float = 0.1
    dynamic_pid_target_error_threshold: float = 4.0
    dynamic_pid_speed_multiplier: float = 1.0
    dynamic_pid_min_coefficient: float = 1.6
    dynamic_pid_max_coefficient: float = 2.7
    dynamic_pid_transition_sharpness: float = 5.0
    dynamic_pid_transition_midpoint: float = 0.0
    dynamic_pid_minimum_data_count: float = 2.0
    dynamic_pid_error_change_tolerance: float = 3.0
    dynamic_pid_smoothing_factor: float = 1.0
    dynamic_pid_aim_ratio: float = 40.0


@dataclass
class ExecutorConfig:
    default: str = "dry_run"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "logs"


@dataclass
class HardwareConfig:
    kind: str = "none"
    host: str = "192.168.2.188"
    port: int = 8888
    uuid: str = "12345678"
    monitor_port: int = 5001
    flip_dy: bool = True
    serial_port: str = ""
    heartbeat_timeout_ms: float = 50.0


@dataclass
class RuntimeConfig:
    web: WebConfig = field(default_factory=WebConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    consumers: ConsumerConfig = field(default_factory=ConsumerConfig)
    limits: RuntimeLimitsConfig = field(default_factory=RuntimeLimitsConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
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
    control = normalized.get("control")
    if isinstance(control, dict):
        control = dict(control)
        for key in (
            "dynamic_pid_fov_deg",
            "dynamic_pid_counts_per_revolution_x",
            "dynamic_pid_counts_per_revolution_y",
            "dynamic_pid_error_mode",
            "dynamic_pid_target_extent_mode",
            "dynamic_pid_y_sign",
            "dynamic_pid_max_x",
            "dynamic_pid_max_y",
        ):
            control.pop(key, None)
        normalized["control"] = control
    return normalized


def _validate_runtime_rules(cfg: RuntimeConfig) -> None:
    if cfg.source.default not in {"null", "capture", "image"} and not cfg.source.default.startswith("image:"):
        raise ValueError("runtime config key 'source.default' must be one of null, capture, image, or image:<path>")
    if cfg.source.image_fps not in {1, 5, 15, 30, 60}:
        raise ValueError("runtime config key 'source.image_fps' must be one of 1, 5, 15, 30, 60")
    if cfg.limits.stream_fps not in {15, 30, 60}:
        raise ValueError("runtime config key 'limits.stream_fps' must be one of 15, 30, 60")
    try:
        cfg.roi.size = normalize_roi_size(cfg.roi.size)
    except ValueError as exc:
        allowed = ", ".join(str(size) for size in ROI_SIZE_CHOICES)
        raise ValueError(f"unsupported ROI size: {cfg.roi.size}; must be one of {allowed}") from exc
    if cfg.roi.mode not in {"center", "manual"}:
        raise ValueError("unsupported ROI mode: must be center or manual")
    if cfg.inference.backend not in {"onnxruntime", "tensorrt"}:
        raise ValueError("runtime config key 'inference.backend' must be onnxruntime or tensorrt")
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
    if cfg.control.strategy not in {"straight", "pid", "proportional", "predictive", "isolated_mouse", "dynamic_pid"}:
        raise ValueError("runtime config key 'control.strategy' must be straight, pid, proportional, predictive, isolated_mouse, or dynamic_pid")
    if cfg.control.fov_ratio <= 0 or cfg.control.fov_ratio > 1:
        raise ValueError("runtime config key 'control.fov_ratio' must be > 0 and <= 1")
    if cfg.control.target_sticky_bias < 0 or cfg.control.target_sticky_bias > 0.9:
        raise ValueError("runtime config key 'control.target_sticky_bias' must be >= 0 and <= 0.9")
    if cfg.control.aim_ratio < 0 or cfg.control.aim_ratio > 100:
        raise ValueError("runtime config key 'control.aim_ratio' must be >= 0 and <= 100")
    cfg.control.trigger_bindings = _normalize_trigger_bindings(cfg.control.trigger_bindings)
    if cfg.control.trigger_mode not in {"hardware", "telemetry", "always"}:
        raise ValueError("runtime config key 'control.trigger_mode' must be hardware, telemetry, or always")
    if cfg.control.target_lost_grace_frames < 0:
        raise ValueError("runtime config key 'control.target_lost_grace_frames' must be >= 0")
    if cfg.control.move_kind not in {"raw", "enc_raw", "auto", "enc_auto", "bezier", "enc_bezier"}:
        raise ValueError("runtime config key 'control.move_kind' must be raw, enc_raw, auto, enc_auto, bezier, or enc_bezier")
    if cfg.control.pid_kd < -1 or cfg.control.pid_kd > 1:
        raise ValueError("runtime config key 'control.pid_kd' must be >= -1 and <= 1")
    for key in (
        "pid_integral_limit",
        "pid_move_limit",
        "kp_x_move_max",
        "kp_y_move_max",
        "prediction_factor",
        "move_ms",
        "trace_ms",
        "command_interval_ms",
        "y_rate_window_ms",
        "y_rate_max_counts",
        "deadzone_counts",
        "near_px",
        "near_speed",
        "far_speed",
        "ema_alpha",
        "counts_per_revolution_x",
        "counts_per_revolution_y",
        "bezier_curvature",
        "straight_c360",
        "straight_kp_x",
        "straight_kp_y",
        "straight_first_frame_gain",
        "straight_first_max_step",
        "straight_max_step",
        "straight_in_deadzone",
        "straight_jump_threshold",
        "straight_pred_gain",
        "straight_pred_consistency_frames",
        "straight_gain_y",
        "isolated_kp_x",
        "isolated_kp_y",
        "isolated_max_x",
        "isolated_max_y",
        "isolated_deadzone_px",
        "isolated_aim_ratio",
        "isolated_smoothing",
        "isolated_prediction",
        "isolated_counts_per_revolution_x",
        "isolated_counts_per_revolution_y",
        "dynamic_pid_kp_x",
        "dynamic_pid_kp_y",
        "dynamic_pid_ki",
        "dynamic_pid_target_error_threshold",
        "dynamic_pid_speed_multiplier",
        "dynamic_pid_min_coefficient",
        "dynamic_pid_max_coefficient",
        "dynamic_pid_transition_sharpness",
        "dynamic_pid_transition_midpoint",
        "dynamic_pid_minimum_data_count",
        "dynamic_pid_error_change_tolerance",
        "dynamic_pid_smoothing_factor",
        "dynamic_pid_aim_ratio",
    ):
        if getattr(cfg.control, key) < 0:
            raise ValueError(f"runtime config key 'control.{key}' must be >= 0")
    if cfg.control.straight_fov_deg <= 0 or cfg.control.straight_fov_deg >= 180:
        raise ValueError("runtime config key 'control.straight_fov_deg' must be > 0 and < 180")
    if cfg.control.ema_alpha > 1:
        raise ValueError("runtime config key 'control.ema_alpha' must be <= 1")
    if cfg.control.prediction_factor > 1:
        raise ValueError("runtime config key 'control.prediction_factor' must be <= 1")
    if cfg.control.isolated_aim_ratio > 100:
        raise ValueError("runtime config key 'control.isolated_aim_ratio' must be <= 100")
    if cfg.control.isolated_smoothing > 0.95:
        raise ValueError("runtime config key 'control.isolated_smoothing' must be <= 0.95")
    if cfg.control.isolated_prediction > 2:
        raise ValueError("runtime config key 'control.isolated_prediction' must be <= 2")
    if cfg.control.isolated_fov_deg <= 0 or cfg.control.isolated_fov_deg >= 180:
        raise ValueError("runtime config key 'control.isolated_fov_deg' must be > 0 and < 180")
    if cfg.control.dynamic_pid_kd < -1 or cfg.control.dynamic_pid_kd > 1:
        raise ValueError("runtime config key 'control.dynamic_pid_kd' must be >= -1 and <= 1")
    if cfg.control.dynamic_pid_smoothing_factor > 1:
        raise ValueError("runtime config key 'control.dynamic_pid_smoothing_factor' must be <= 1")
    if cfg.control.dynamic_pid_aim_ratio > 100:
        raise ValueError("runtime config key 'control.dynamic_pid_aim_ratio' must be <= 100")

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


def _normalize_trigger_bindings(value: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) >= 2:
            break
    return result


def save_runtime_config(cfg: RuntimeConfig, path: str | Path) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
