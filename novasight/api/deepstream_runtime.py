from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import HTTPException, Request

from novasight.deepstream import DeepStreamDetectionBackend, DeepStreamPipelineConfig
from novasight.model_registry.deepstream_config import (
    nvinfer_config_fingerprint,
    read_nvinfer_config_fingerprint,
    validate_nvinfer_config_engine_path,
    validate_nvinfer_config_properties,
)
from novasight.model_registry.manifest import (
    ModelManifest,
    read_manifest,
    validate_manifest_engine_artifact,
)
from novasight.roi import center_roi_region


def build_deepstream_detection_source(request: Request) -> DeepStreamDetectionBackend:
    config = request.app.state.runtime.config
    manifest_path, nvinfer_config_path = deepstream_artifact_paths(request)
    if not manifest_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"DeepStream manifest not found: {manifest_path}",
        )
    if not nvinfer_config_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"DeepStream nvinfer config not found: {nvinfer_config_path}",
        )
    try:
        manifest = read_manifest(manifest_path)
        engine_path = manifest_path.with_name(manifest.artifact.engine_path)
        validate_manifest_engine_artifact(manifest, engine_path)
        validate_deepstream_config_matches_manifest(
            manifest,
            nvinfer_config_path=nvinfer_config_path,
            engine_path=engine_path,
        )
        model_height, model_width = manifest_input_hw(manifest)
        capture_width = int(config.capture.width or 1920)
        capture_height = int(config.capture.height or 1080)
        fps = int(config.capture.fps or 120)
        roi_left, roi_top, roi_size = center_roi_region(
            source_width=capture_width,
            source_height=capture_height,
            requested_size=config.roi.size,
            offset_x=config.roi.offset_x,
            offset_y=config.roi.offset_y,
        )
        pipeline_config = DeepStreamPipelineConfig(
            device=config.capture.device,
            capture_width=capture_width,
            capture_height=capture_height,
            fps=fps,
            roi_left=roi_left,
            roi_top=roi_top,
            roi_size=roi_size,
            model_width=model_width,
            model_height=model_height,
            nvinfer_config_path=nvinfer_config_path,
            pixel_format=str(config.capture.pixel_format or "MJPG"),
            io_mode=int(getattr(config.inference, "deepstream_io_mode", 2)),
            batched_push_timeout_us=int(
                getattr(config.inference, "deepstream_batched_push_timeout_us", 0)
            ),
            tracker_config_path=_optional_path(
                getattr(config.inference, "deepstream_tracker_config_path", "")
            ),
        )
        return DeepStreamDetectionBackend(
            pipeline_config=pipeline_config,
            manifest=manifest,
            roi_width=roi_size,
            roi_height=roi_size,
            confidence_threshold=float(config.inference.confidence_threshold),
            nms_threshold=float(config.inference.nms_threshold),
            max_publish_age_ms=float(config.control.latency_reject_if_age_exceeds_ms),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def validate_deepstream_config_matches_manifest(
    manifest: ModelManifest,
    *,
    nvinfer_config_path: Path,
    engine_path: Path,
) -> None:
    expected = nvinfer_config_fingerprint(manifest)
    actual = read_nvinfer_config_fingerprint(nvinfer_config_path)
    if actual != expected:
        detail = "missing" if not actual else "mismatch"
        raise ValueError(
            "DeepStream nvinfer config does not match model manifest "
            f"({detail}); prepare DeepStream config again"
        )
    validate_nvinfer_config_engine_path(nvinfer_config_path, engine_path)
    validate_nvinfer_config_properties(nvinfer_config_path, manifest)


def deepstream_artifact_paths(request: Request) -> tuple[Path, Path]:
    config = request.app.state.runtime.config
    inference = config.inference
    manifest_path = _configured_path(
        request,
        str(getattr(inference, "deepstream_manifest_path", "") or ""),
    )
    nvinfer_config_path = _configured_path(
        request,
        str(getattr(inference, "deepstream_config_path", "") or ""),
    )
    if manifest_path is not None and nvinfer_config_path is not None:
        return manifest_path, nvinfer_config_path
    active_dir = active_deployment_dir(request)
    return (
        manifest_path or active_dir / "model.manifest.json",
        nvinfer_config_path or active_dir / "deepstream.ini",
    )


def manifest_input_hw(manifest: ModelManifest) -> tuple[int, int]:
    shape = list(manifest.input.shape)
    layout = str(manifest.input.layout).upper()
    if layout != "NCHW" or len(shape) != 4:
        raise ValueError(
            f"DeepStream backend expects NCHW input shape, got {layout} {shape}"
        )
    manifest_batch = int(shape[0])
    runtime_batch = int(getattr(manifest.runtime, "batch_size", 1))
    if manifest_batch != 1 or runtime_batch != 1:
        raise ValueError(
            "DeepStream runtime currently supports single-source batch_size=1 "
            f"(input batch={manifest_batch}, runtime batch_size={runtime_batch})"
        )
    height = int(shape[2])
    width = int(shape[3])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid DeepStream input shape: {shape}")
    return height, width


def _optional_path(value: object) -> Path | None:
    text = str(value or "").strip()
    return Path(text).expanduser() if text else None


def active_deployment_dir(request: Request) -> Path:
    models = request.app.state.models
    deployment = models.get_active_deployment()
    if deployment is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "DeepStream backend requires an active model deployment "
                "or explicit manifest/config paths"
            ),
        )
    artifact = models.get_artifact(deployment.artifact_id)
    if artifact is None:
        raise HTTPException(
            status_code=400,
            detail="active deployment artifact not found",
        )
    if artifact.kind != "engine":
        raise HTTPException(
            status_code=400,
            detail="DeepStream backend requires an engine artifact",
        )
    version = models.get_version(artifact.version_id)
    if version is None:
        raise HTTPException(
            status_code=400,
            detail="active deployment version not found",
        )
    project = models.get_project(version.project_id)
    if project is None:
        raise HTTPException(
            status_code=400,
            detail="active deployment project not found",
        )
    return Path(models.data_dir) / project.name / version.version


def _configured_path(request: Request, value: str) -> Path | None:
    text = unquote(value.strip())
    if not text:
        return None
    models_root = Path(request.app.state.models.data_dir).expanduser().resolve(strict=False)
    path = Path(text).expanduser()
    if path.is_absolute():
        resolved = path.resolve(strict=False)
    else:
        registry_prefix = Path(models_root.name)
        parent_prefix = Path(models_root.parent.name) / models_root.name
        if path.parts[: len(parent_prefix.parts)] == parent_prefix.parts:
            path = Path(*path.parts[len(parent_prefix.parts) :])
        elif path.parts[: len(registry_prefix.parts)] == registry_prefix.parts:
            path = Path(*path.parts[len(registry_prefix.parts) :])
        resolved = (models_root / path).resolve(strict=False)
    try:
        resolved.relative_to(models_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"DeepStream artifact path must stay inside model registry: {text}",
        ) from exc
    return resolved
