from __future__ import annotations

from pathlib import Path

from novasight.model_registry.deepstream_config import (
    generate_nvinfer_config as _generate_registry_nvinfer_config,
    nvinfer_config_fingerprint,
    read_nvinfer_config_fingerprint,
    read_nvinfer_config_property,
    validate_nvinfer_config_engine_path,
    validate_nvinfer_config_properties,
)
from novasight.model_registry.manifest import ModelManifest


def generate_nvinfer_config(
    manifest: ModelManifest,
    *,
    engine_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> str:
    config_text, _fingerprint = generate_nvinfer_config_with_fingerprint(
        manifest,
        engine_path=engine_path,
        manifest_path=manifest_path,
    )
    return config_text


def generate_nvinfer_config_with_fingerprint(
    manifest: ModelManifest,
    *,
    engine_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[str, str]:
    resolved_engine_path = resolve_engine_path(
        manifest,
        engine_path=engine_path,
        manifest_path=manifest_path,
    )
    return _generate_registry_nvinfer_config(manifest, engine_path=resolved_engine_path)


def resolve_engine_path(
    manifest: ModelManifest,
    *,
    engine_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    if engine_path is not None:
        return Path(engine_path).expanduser().resolve(strict=False)
    artifact_name = str(manifest.artifact.engine_path or "").strip()
    if not artifact_name:
        raise ValueError("model manifest artifact.engine_path is required")
    base_dir = Path.cwd()
    if manifest_path is not None:
        base_dir = Path(manifest_path).expanduser().resolve(strict=False).parent
    return (base_dir / artifact_name).resolve(strict=False)


__all__ = [
    "generate_nvinfer_config",
    "generate_nvinfer_config_with_fingerprint",
    "nvinfer_config_fingerprint",
    "read_nvinfer_config_fingerprint",
    "read_nvinfer_config_property",
    "resolve_engine_path",
    "validate_nvinfer_config_engine_path",
    "validate_nvinfer_config_properties",
]
