from __future__ import annotations

from dataclasses import MISSING, Field, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 5174


@dataclass
class SourceConfig:
    default: str = "null"
    target_fps: int = 60


@dataclass
class RuntimeLimitsConfig:
    max_frame_queue: int = 2
    stream_fps: int = 30


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
    max_abs_dx: int = 120
    max_abs_dy: int = 120
    min_confidence: float = 0.0
    output_mode: str = ""


@dataclass
class ExecutorConfig:
    default: str = "dry_run"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class RuntimeConfig:
    web: WebConfig = field(default_factory=WebConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    limits: RuntimeLimitsConfig = field(default_factory=RuntimeLimitsConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


T = TypeVar("T")


def _field_default(item: Field[Any]) -> Any:
    if item.default_factory is not MISSING:
        return item.default_factory()
    if item.default is not MISSING:
        return item.default
    return None


def _validate_leaf_value(key_name: str, value: Any, expected_type: type[Any]) -> None:
    if expected_type is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"runtime config key '{key_name}' must be an int")
    elif expected_type is float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"runtime config key '{key_name}' must be a number")
    elif expected_type is str:
        if not isinstance(value, str):
            raise ValueError(f"runtime config key '{key_name}' must be a string")


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


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return RuntimeConfig()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {cfg_path}")
    return _build_dataclass(RuntimeConfig, raw)


def save_runtime_config(cfg: RuntimeConfig, path: str | Path) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
