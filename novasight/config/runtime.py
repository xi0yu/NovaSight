from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

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
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


T = TypeVar("T")


def _build_dataclass(cls: type[T], raw: dict[str, Any]) -> T:
    values: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in raw:
            continue
        current = getattr(cls(), item.name) if callable(cls) else None
        value = raw[item.name]
        if is_dataclass(current) and isinstance(value, dict):
            values[item.name] = _build_dataclass(type(current), value)
        else:
            values[item.name] = value
    return cls(**values)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return RuntimeConfig()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
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
