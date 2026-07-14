from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from novasight.model_registry import (
    ModelArtifactScanResult,
    ModelRegistry,
    RegistryConflictError,
    RegistryError,
    RegistryNotFoundError,
    RegistryValidationError,
    inspect_model_artifact,
    scan_model_artifacts,
)
from novasight.inference import parse_tensor_input_shape
from novasight.deepstream.model_manifest import (
    recommend_engine_manifest,
)
from novasight.api.routes_model_ingress import (
    load_validated_profile,
    serialized_model_operation,
    set_profile_activation,
)
from novasight.model_ingress import InferenceConfigBuilder, ModelProfileStore

router = APIRouter(prefix="/api/models")
logger = logging.getLogger("novasight.api.models")
_ENGINE_INPUT_SHAPE_PENDING = "engine-probe-required"


YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLOV8N_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class ProjectCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    description: StrictStr = ""


class VersionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: StrictStr
    source_kind: StrictStr
    source_path: StrictStr
    classes: list[StrictStr]
    input_shape: StrictStr


class ArtifactCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: StrictStr
    path: StrictStr
    checksum: StrictStr
    status: StrictStr


class CatalogRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: StrictStr


class ConversionJobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_kind: StrictStr
    command: list[StrictStr]


class ConversionJobFinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StrictStr
    log: StrictStr = ""


class ConversionJobListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: StrictInt | None = None


class PublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: StrictInt


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _find_project_by_name(registry: ModelRegistry, name: str):
    return next(
        (project for project in registry.list_projects() if project.name == name),
        None,
    )


def _find_version(registry: ModelRegistry, project_id: int, version_name: str):
    return next(
        (
            version
            for version in registry.list_versions(project_id)
            if version.version == version_name
        ),
        None,
    )


def _find_artifact(registry: ModelRegistry, version_id: int, path: str):
    return next(
        (
            artifact
            for artifact in registry.list_artifacts(version_id)
            if artifact.path == path
        ),
        None,
    )


def _find_artifact_by_checksum(
    registry: ModelRegistry,
    *,
    project_id: int,
    kind: str,
    checksum: str,
    classes: list[str],
    input_shape: str,
):
    for version in registry.list_versions(project_id):
        if version.classes != classes or version.input_shape != input_shape:
            continue
        for artifact in registry.list_artifacts(version.id):
            if artifact.kind == kind and artifact.checksum == checksum:
                return version, artifact
    return None, None


def _checksum_token(checksum: str) -> str:
    value = checksum.split(":", 1)[-1].strip().lower()
    return value[:12] or "unknown"


def _immutable_artifact_filename(filename: str, checksum: str) -> str:
    path = Path(filename)
    immutable_name = f"{path.stem}.{_checksum_token(checksum)}{path.suffix.lower()}"
    if path.parent == Path("."):
        return immutable_name
    return (path.parent / immutable_name).as_posix()


def _version_content_token(
    checksum: str,
    *,
    classes: list[str],
    input_shape: str,
) -> str:
    metadata = json.dumps(
        {"classes": classes, "input_shape": input_shape},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    metadata_token = hashlib.sha256(metadata.encode("utf-8")).hexdigest()[:8]
    return f"{_checksum_token(checksum)}-{metadata_token}"


def _select_artifact_version(
    registry: ModelRegistry,
    *,
    project: Any,
    requested_version_name: str,
    kind: str,
    source_path: str,
    classes: list[str],
    input_shape: str,
    checksum: str,
):
    matched_version, matched_artifact = _find_artifact_by_checksum(
        registry,
        project_id=project.id,
        kind=kind,
        checksum=checksum,
        classes=classes,
        input_shape=input_shape,
    )
    if matched_version is not None:
        return matched_version, matched_artifact

    base_version = _find_version(registry, project.id, requested_version_name)
    if base_version is None:
        version_name = requested_version_name
    else:
        same_kind = any(
            artifact.kind == kind
            for artifact in registry.list_artifacts(base_version.id)
        )
        if not same_kind:
            return base_version, None
        content_token = _version_content_token(
            checksum,
            classes=classes,
            input_shape=input_shape,
        )
        version_name = _safe_component(
            f"{requested_version_name}-{content_token}",
            content_token,
        )
        existing = _find_version(registry, project.id, version_name)
        if existing is not None:
            return existing, None

    return (
        registry.create_version(
            project_id=project.id,
            version=version_name,
            source_kind=_source_kind_from_artifact(kind),
            source_path=source_path,
            classes=classes,
            input_shape=input_shape,
        ),
        None,
    )


def _artifact_kind_from_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pt":
        return "pt"
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".engine":
        return "engine"
    raise RegistryValidationError("model file must end with .pt, .onnx, or .engine")


def _source_kind_from_artifact(kind: str) -> str:
    if kind in {"pt", "onnx"}:
        return kind
    return "onnx"


def _parse_classes(value: str) -> list[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    return classes or ["target"]


def _normalize_input_shape(value: Any, fallback: str = "1x3x640x640") -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    try:
        return str(parse_tensor_input_shape(raw))
    except ValueError:
        return fallback


def _shape_from_dims(dims: list[int | None]) -> str | None:
    if len(dims) == 2:
        height = dims[0] or 640
        width = dims[1] or 640
        return str(parse_tensor_input_shape(f"{height}x{width}"))
    if len(dims) == 3:
        channels = dims[0] or 3
        height = dims[1] or 640
        width = dims[2] or 640
        return str(parse_tensor_input_shape(f"{channels}x{height}x{width}"))
    if len(dims) == 4:
        batch = dims[0] or 1
        channels = dims[1] or 3
        height = dims[2] or 640
        width = dims[3] or 640
        return str(parse_tensor_input_shape(f"{batch}x{channels}x{height}x{width}"))
    return None


def _detect_onnx_input_shape(model_file: Path) -> str | None:
    try:
        import onnx

        model = onnx.load(str(model_file))
        initializers = {item.name for item in model.graph.initializer}
        graph_input = next(
            (item for item in model.graph.input if item.name not in initializers),
            model.graph.input[0] if model.graph.input else None,
        )
        if graph_input is None:
            return None
        shape = graph_input.type.tensor_type.shape
        dims: list[int | None] = []
        for dim in shape.dim:
            dims.append(int(dim.dim_value) if dim.dim_value > 0 else None)
        return _shape_from_dims(dims)
    except ModuleNotFoundError as exc:
        if exc.name == "onnx":
            logger.debug("ONNX input shape auto-detect unavailable path=%s error=%s", model_file, exc)
            return None
        logger.info("ONNX input shape auto-detect skipped path=%s error=%s", model_file, exc)
        return None
    except Exception as exc:
        logger.info("ONNX input shape auto-detect skipped path=%s error=%s", model_file, exc)
        return None


def _detect_input_shape_from_filename(model_file: Path) -> str | None:
    match = re.search(r"(?<!\d)(\d{2,5})[xX](\d{2,5})(?!\d)", model_file.stem)
    if match is None:
        return None
    height = int(match.group(1))
    width = int(match.group(2))
    try:
        return str(parse_tensor_input_shape(f"{height}x{width}"))
    except ValueError:
        return None


def _detect_input_shape(model_file: Path, sidecar: dict[str, Any]) -> str:
    if model_file.suffix.lower() == ".engine":
        return _ENGINE_INPUT_SHAPE_PENDING
    if "input_shape" in sidecar:
        return _normalize_input_shape(sidecar.get("input_shape"))
    detected_from_name = _detect_input_shape_from_filename(model_file)
    if detected_from_name:
        return detected_from_name
    if model_file.suffix.lower() == ".onnx":
        detected = _detect_onnx_input_shape(model_file)
        if detected:
            return detected
    return "1x3x640x640"


def _safe_component(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return normalized or fallback


def _model_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _deepstream_model_defaults(
    artifact_path: Path,
    source_path: str,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    sidecar_paths = [artifact_path.with_suffix(".json")]
    source_text = str(source_path or "").strip()
    if source_text:
        source = Path(source_text)
        sidecar_paths.append(source.with_suffix(".json"))
    sidecar: dict[str, Any] = {}
    sidecar_path: Path | None = None
    for candidate in sidecar_paths:
        loaded = _model_sidecar(candidate)
        if loaded:
            sidecar = loaded
            sidecar_path = candidate
            break

    filename = artifact_path.name.lower()
    inferred_precision = (
        "int8"
        if "int8" in filename
        else "fp32"
        if "fp32" in filename
        else "fp16"
    )
    defaults = {
        "runtime_precision": str(
            sidecar.get("runtime_precision")
            or sidecar.get("precision")
            or inferred_precision
        ),
        "input_color_format": str(sidecar.get("input_color_format") or "RGB").upper(),
        "input_scale_factor": float(sidecar.get("input_scale_factor", 1.0 / 255.0)),
        "maintain_aspect_ratio": bool(sidecar.get("maintain_aspect_ratio", False)),
        "symmetric_padding": bool(sidecar.get("symmetric_padding", False)),
    }
    source_label = (
        f"model_sidecar:{sidecar_path}"
        if sidecar_path is not None
        else "novasight_yolo_default"
    )
    sources = {
        "runtime_precision": source_label if sidecar_path is not None else "engine_filename_or_fp16_default",
        "input_color_format": source_label,
        "input_scale_factor": source_label,
        "maintain_aspect_ratio": source_label,
        "symmetric_padding": source_label,
    }
    warnings: list[str] = []
    if sidecar_path is None:
        warnings.append(
            "TensorRT engine 不保存 RGB/BGR、归一化和 letterbox 语义；当前使用 NovaSight YOLO 默认值。"
        )
        warnings.append(
            "如模型预处理不同，请在同名 .json sidecar 中提供 input_color_format、input_scale_factor、maintain_aspect_ratio 和 symmetric_padding。"
        )
    return defaults, sources, warnings


def _relative_registry_path(registry: ModelRegistry, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return Path(path).resolve(strict=False).relative_to(
            Path(registry.data_dir).resolve(strict=False)
        ).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _registry_status_from_scan(status: str) -> str:
    if status == "ready":
        return "ready"
    if status == "need_confirm":
        return "pending"
    return "failed"


def _artifact_registry_status(kind: str, artifact_path: Path) -> str:
    if str(kind).lower() != "engine":
        return "ready"
    return _registry_status_from_scan(inspect_model_artifact(artifact_path).status)


def _artifact_scan_payload(
    registry: ModelRegistry,
    result: ModelArtifactScanResult,
) -> dict[str, Any]:
    return {
        "path": _relative_registry_path(registry, result.path),
        "kind": result.kind,
        "status": result.status,
        "reason": result.reason,
        "sha256": result.sha256,
        "size_bytes": result.size_bytes,
        "manifest_path": _relative_registry_path(registry, result.manifest_path),
        "model_fingerprint": result.model_fingerprint,
    }


def _artifact_scan_counts(results: list[ModelArtifactScanResult]) -> dict[str, int]:
    counts = {
        "ready": 0,
        "need_confirm": 0,
        "invalid": 0,
        "unsupported": 0,
    }
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _model_catalog_registry_indexes(
    registry: ModelRegistry,
) -> dict[Path, dict[str, Any]]:
    by_path: dict[Path, dict[str, Any]] = {}
    for project in registry.list_projects():
        for version in registry.list_versions(project.id):
            for artifact in registry.list_artifacts(version.id):
                entry = {
                    "project_id": project.id,
                    "project_name": project.name,
                    "version_id": version.id,
                    "version_name": version.version,
                    "artifact_id": artifact.id,
                    "artifact_status": artifact.status,
                }
                artifact_path = registry.resolve_artifact_path(artifact)
                by_path[artifact_path] = entry
    return by_path


def _model_catalog_payload(
    registry: ModelRegistry,
    results: list[ModelArtifactScanResult],
    *,
    catalog_roots: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    roots = catalog_roots or (Path(registry.data_dir),)
    resolved_roots = tuple(Path(root).resolve(strict=False) for root in roots)
    registry_by_path = _model_catalog_registry_indexes(registry)
    root: dict[str, Any] = {
        "type": "directory",
        "name": "models",
        "relative_path": "",
        "children": [],
    }
    directories: dict[str, dict[str, Any]] = {"": root}
    model_count = 0
    seen_artifact_ids: set[int] = set()

    for result in sorted(
        results,
        key=lambda item: (
            0
            if item.path.resolve(strict=False) in registry_by_path
            else 1,
            len(item.path.resolve(strict=False).parts),
            item.path.as_posix().lower(),
        ),
    ):
        relative = _catalog_relative_path(result.path, resolved_roots)
        if relative is None:
            continue
        registry_entry = registry_by_path.get(result.path.resolve(strict=False))
        artifact_id = registry_entry.get("artifact_id") if registry_entry else None
        if isinstance(artifact_id, int):
            if artifact_id in seen_artifact_ids:
                continue
            seen_artifact_ids.add(artifact_id)
        parent = root
        parent_parts: list[str] = []
        for part in relative.parts[:-1]:
            parent_parts.append(part)
            directory_path = Path(*parent_parts).as_posix()
            directory = directories.get(directory_path)
            if directory is None:
                directory = {
                    "type": "directory",
                    "name": part,
                    "relative_path": directory_path,
                    "children": [],
                }
                directories[directory_path] = directory
                parent["children"].append(directory)
            parent = directory

        parent["children"].append(
            {
                "type": "model",
                "name": relative.name,
                "relative_path": relative.as_posix(),
                "kind": result.kind,
                "size_bytes": result.size_bytes,
                "scan_status": result.status,
                "scan_reason": result.reason,
                **(registry_entry or {}),
            }
        )
        model_count += 1

    def sort_children(node: dict[str, Any]) -> None:
        children = node.get("children", [])
        children.sort(
            key=lambda child: (
                0 if child.get("type") == "directory" else 1,
                str(child.get("name", "")).lower(),
            )
        )
        for child in children:
            if child.get("type") == "directory":
                sort_children(child)

    sort_children(root)
    return {
        "root": root,
        "directory_count": max(0, len(directories) - 1),
        "model_count": model_count,
    }


def _catalog_relative_path(path: Path, roots: tuple[Path, ...]) -> Path | None:
    resolved = Path(path).resolve(strict=False)
    for root in roots:
        try:
            return resolved.relative_to(root)
        except ValueError:
            continue
    return None


def _resolve_catalog_engine_path(
    registry: ModelRegistry,
    relative_path: str,
) -> Path:
    requested = Path(str(relative_path).strip())
    if not requested.parts or requested.is_absolute() or ".." in requested.parts:
        raise RegistryValidationError("catalog model path must be a relative path")
    for root in (Path("models"), Path(registry.data_dir)):
        resolved_root = root.resolve(strict=False)
        candidate = (resolved_root / requested).resolve(strict=False)
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue
        if candidate.is_file():
            if candidate.suffix.lower() != ".engine":
                raise RegistryValidationError(
                    "catalog registration currently supports TensorRT .engine files only"
                )
            return candidate
    raise RegistryNotFoundError(f"catalog model does not exist: {relative_path}")


def _find_registered_artifact_by_path(
    registry: ModelRegistry,
    engine_path: Path,
):
    expected = engine_path.resolve(strict=False)
    for project in registry.list_projects():
        for version in registry.list_versions(project.id):
            for artifact in registry.list_artifacts(version.id):
                if registry.resolve_artifact_path(artifact) == expected:
                    return project, version, artifact
    return None


def _artifact_asset_context(
    registry: ModelRegistry,
    artifact_id: int,
) -> tuple[Any, Any, Any, Path]:
    artifact = registry.get_artifact(artifact_id)
    if artifact is None:
        raise RegistryNotFoundError(f"unknown artifact id: {artifact_id}")
    version = registry.get_version(artifact.version_id)
    if version is None:
        raise RegistryNotFoundError(f"unknown version id: {artifact.version_id}")
    project = registry.get_project(version.project_id)
    if project is None:
        raise RegistryNotFoundError(f"unknown project id: {version.project_id}")
    artifact_path = registry.resolve_artifact_path(artifact)
    return artifact, version, project, artifact_path


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.models


def _as_http_error(exc: RegistryError) -> HTTPException:
    if isinstance(exc, RegistryNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (RegistryConflictError, RegistryValidationError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def _require_project(registry: ModelRegistry, project_id: int) -> None:
    if registry.get_project(project_id) is None:
        raise RegistryNotFoundError(f"unknown project id: {project_id}")


def _require_version(registry: ModelRegistry, version_id: int) -> None:
    if registry.get_version(version_id) is None:
        raise RegistryNotFoundError(f"unknown version id: {version_id}")


def _load_published_artifact(
    request: Request,
    registry: ModelRegistry,
    artifact_id: int,
) -> None:
    artifact = registry.get_artifact(artifact_id)
    if artifact is None:
        return
    if artifact.kind not in {"onnx", "engine"}:
        request.app.state.inference.unload(
            f"published artifact is not runnable inference artifact: {artifact.kind}"
        )
        return
    version = registry.get_version(artifact.version_id)
    if version is None:
        raise RegistryNotFoundError(f"unknown version id: {artifact.version_id}")
    project = registry.get_project(version.project_id)
    if project is None:
        raise RegistryNotFoundError(f"unknown project id: {version.project_id}")
    artifact_path = registry.resolve_artifact_path(artifact)
    try:
        profile = load_validated_profile(artifact_path)
    except ValueError as exc:
        raise RegistryValidationError(str(exc)) from exc
    if _deepstream_selected(request):
        inference_config = request.app.state.config.inference
        try:
            runtime_config = InferenceConfigBuilder().build(
                profile,
                parser_library_path=Path(inference_config.deepstream_parser_library),
            )
        except ValueError as exc:
            raise RegistryValidationError(str(exc)) from exc
        ModelProfileStore().write(
            profile,
            runtime_manifest=runtime_config.manifest,
        )
        request.app.state.inference.unload(
            "TensorRT engine ownership delegated to DeepStream nvinfer"
        )
        _clear_runtime_pipeline_after_model_switch(
            request,
            "DeepStream active model changed",
        )
        return
    candidate, status = request.app.state.inference.prepare_profile(
        profile,
        diagnostic=False,
    )
    if status.get("loaded") is not True or status.get("warmed") is not True:
        _close_candidate(candidate)
        raise RegistryValidationError(
            str(status.get("reason") or "validated model failed to load")
        )
    request.app.state.inference.commit(
        candidate,
        artifact_path=artifact_path,
        classes=list(profile.labels),
        input_shape="x".join(str(value) for value in profile.input.runtime_shape),
    )


def _resolve_runnable_artifact(
    registry: ModelRegistry,
    *,
    project_id: int,
    artifact_id: int,
) -> tuple[Path, list[str], str]:
    artifact = registry.get_artifact(artifact_id)
    if artifact is None:
        raise RegistryNotFoundError(f"unknown artifact id: {artifact_id}")
    if artifact.status not in {"ready", "pending"}:
        raise RegistryValidationError(
            f"artifact status does not allow safe validation: {artifact.status}"
        )
    if artifact.kind not in {"onnx", "engine"}:
        raise RegistryValidationError(
            f"published artifact is not runnable inference artifact: {artifact.kind}"
        )
    version = registry.get_version(artifact.version_id)
    if version is None:
        raise RegistryNotFoundError(f"unknown version id: {artifact.version_id}")
    if version.project_id != project_id:
        raise RegistryValidationError("artifact does not belong to project")
    project = registry.get_project(version.project_id)
    if project is None:
        raise RegistryNotFoundError(f"unknown project id: {version.project_id}")
    return (
        registry.resolve_artifact_path(artifact),
        list(version.classes),
        version.input_shape,
    )


def _prepare_runnable_artifact(
    request: Request,
    *,
    artifact_path: Path,
    classes: list[str],
    input_shape: str,
) -> tuple[Any, dict[str, Any]]:
    try:
        profile = load_validated_profile(artifact_path)
    except ValueError as exc:
        raise RegistryValidationError(str(exc)) from exc
    if not _deepstream_selected(request):
        raise RegistryValidationError(
            "runtime model activation only supports deepstream_nvinfer"
        )
    inference_config = request.app.state.config.inference
    try:
        runtime_config = InferenceConfigBuilder().build(
            profile,
            parser_library_path=Path(inference_config.deepstream_parser_library),
        )
    except ValueError as exc:
        raise RegistryValidationError(str(exc)) from exc
    manifest = runtime_config.manifest
    ModelProfileStore().write(profile, runtime_manifest=manifest)
    return None, {
        "selected": "deepstream_nvinfer",
        "available": True,
        "loaded": True,
        "input_shape": "x".join(str(value) for value in manifest.input.shape),
        "classes": list(manifest.output.class_names),
        "model_fingerprint": manifest.model_fingerprint,
        "reason": "validated for pipeline-owned nvinfer loading",
    }


def _close_candidate(candidate: Any) -> None:
    close = getattr(candidate, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _actual_runtime_input_shape(status: dict[str, Any], fallback: str) -> str:
    actual = status.get("input_shape")
    return str(actual).strip() if isinstance(actual, str) and actual.strip() else fallback


def _set_previous_profile_inactive(
    registry: ModelRegistry,
    previous_artifact_id: int | None,
    active_artifact_id: int,
) -> None:
    if previous_artifact_id is None or previous_artifact_id == active_artifact_id:
        return
    try:
        _artifact, _version, _project, previous_path = _artifact_asset_context(
            registry,
            previous_artifact_id,
        )
        set_profile_activation(previous_path, active=False)
    except (RegistryError, OSError, ValueError) as exc:
        logger.warning(
            "previous ModelProfile activation state could not be cleared artifact_id=%s: %s",
            previous_artifact_id,
            exc,
        )


def _sync_artifact_version_input_shape(
    registry: ModelRegistry,
    *,
    artifact_id: int,
    input_shape: str,
) -> None:
    artifact = registry.get_artifact(artifact_id)
    if artifact is None:
        return
    version = registry.get_version(artifact.version_id)
    if version is None or version.input_shape == input_shape:
        return
    registry.update_version_input_shape(version.id, input_shape)
    logger.info(
        "model version input shape updated from runtime artifact_id=%s version_id=%s input=%s",
        artifact_id,
        version.id,
        input_shape,
    )


def _pause_runtime_pipeline_for_model_switch(request: Request) -> bool:
    runtime = getattr(request.app.state, "runtime", None)
    pipeline = getattr(runtime, "pipeline", None)
    was_running = bool(pipeline is not None and getattr(pipeline, "running", False) is True)
    if was_running:
        logger.info("pausing runtime pipeline for model switch")
        pipeline.stop()
    elif runtime is not None:
        cancel_control = getattr(runtime, "cancel_control", None)
        if callable(cancel_control):
            cancel_control("MODEL_SWITCH")
    capture = getattr(request.app.state, "capture", None)
    latest = getattr(capture, "latest_frame_broker", None)
    clear_latest = getattr(latest, "clear", None)
    if callable(clear_latest):
        clear_latest()
    return was_running


def _resume_runtime_pipeline_after_model_switch(request: Request, should_resume: bool) -> None:
    if not should_resume:
        return
    runtime = getattr(request.app.state, "runtime", None)
    pipeline = getattr(runtime, "pipeline", None)
    if pipeline is None:
        from novasight.runtime.pipeline_factory import create_runtime_pipeline

        try:
            pipeline = create_runtime_pipeline(
                capture=request.app.state.capture,
                runtime=runtime,
            )
            runtime.pipeline = pipeline
        except Exception as exc:
            logger.warning("runtime pipeline rebuild after model switch failed: %s", exc)
            return
    try:
        pipeline.start()
        logger.info("runtime pipeline resumed after model switch")
    except Exception as exc:
        logger.warning("runtime pipeline resume after model switch failed: %s", exc)


def _clear_runtime_pipeline_after_model_switch(request: Request, reason: str) -> None:
    runtime = getattr(request.app.state, "runtime", None)
    pipeline = getattr(runtime, "pipeline", None) if runtime is not None else None
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception as exc:
            logger.warning("runtime pipeline stop after model switch failed: %s", exc)
    if runtime is not None:
        runtime.pipeline = None
        runtime.running = False
    logger.info("runtime pipeline cleared after model switch: %s", reason)


def _inference_status(request: Request) -> dict[str, Any]:
    runtime = getattr(request.app.state, "runtime", None)
    if _deepstream_selected(request) and getattr(runtime, "pipeline", None) is not None:
        status = runtime.pipeline.status()
        deepstream = status.get("deepstream", {}) if isinstance(status, dict) else {}
        if isinstance(deepstream, dict) and deepstream:
            return dict(deepstream)
    if _deepstream_selected(request):
        registry = getattr(request.app.state, "models", None)
        deployment = registry.get_active_deployment() if registry is not None else None
        return {
            "selected": "deepstream_nvinfer",
            "available": deployment is not None,
            "loaded": False,
            "configured": deployment is not None,
            "reason": "validated deployment will be loaded by nvinfer at runtime start",
        }
    inference = getattr(request.app.state, "inference", None)
    status = getattr(inference, "status", None)
    if not callable(status):
        return {"available": False, "loaded": False, "reason": "inference runtime unavailable"}
    return dict(status())


def _deepstream_selected(request: Request) -> bool:
    config = getattr(request.app.state, "config", None)
    inference = getattr(config, "inference", None)
    return str(getattr(inference, "backend", "")).lower() == "deepstream_nvinfer"


def _require_control_inference_policy(request: Request) -> None:
    config = getattr(request.app.state, "config", None)
    inference = getattr(config, "inference", None)
    if inference is None:
        return
    if not bool(getattr(inference, "require_gpu", True)):
        raise RegistryValidationError("formal model activation requires inference.require_gpu=true")
    if bool(getattr(inference, "allow_cpu_fallback", False)):
        raise RegistryValidationError(
            "formal model activation requires inference.allow_cpu_fallback=false"
        )


def _model_switch_report(
    *,
    action: str,
    deployment: Any,
    artifact_path: Path,
    classes: list[str],
    input_shape: str,
    inference_status: dict[str, Any],
) -> dict[str, Any]:
    loaded = inference_status.get("loaded") is True
    selected = str(inference_status.get("selected") or inference_status.get("engine") or "auto")
    configured_for_deepstream = (
        selected == "deepstream_nvinfer" and inference_status.get("configured") is True
    )
    applied = loaded or configured_for_deepstream
    message = (
        f"模型已切换：{artifact_path.name} · {selected} · {input_shape}"
        if applied
        else f"模型登记完成，但推理运行态未加载：{artifact_path.name}"
    )
    return {
        "action": action,
        "applied": applied,
        "rolled_back": False,
        "message": message,
        "artifact_id": deployment.artifact_id,
        "previous_artifact_id": deployment.previous_artifact_id,
        "artifact_path": str(artifact_path),
        "backend": selected,
        "input_shape": input_shape,
        "classes": len(classes),
        "sections": [
            {
                "section": "模型产物",
                "impact": "推理入口",
                "status": "applied",
                "message": f"使用 {artifact_path.suffix.lower() or 'unknown'} 后缀自动选择后端",
            },
            {
                "section": "推理运行态",
                "impact": "采集 -> 推理 -> 控制",
                "status": "applied" if applied else "failed",
                "message": (
                    "候选模型已加载并替换当前模型"
                    if loaded
                    else "模型契约已确认；nvinfer 将在运行时启动时加载"
                    if configured_for_deepstream
                    else str(inference_status.get("reason") or "未加载")
                ),
            },
        ],
    }


@router.get("/projects")
def list_projects(request: Request) -> list[dict[str, Any]]:
    registry = _registry(request)
    return [asdict(project) for project in registry.list_projects()]


def _read_model_catalog(
    registry: ModelRegistry,
    *,
    force: bool = False,
) -> dict[str, Any]:
    catalog_roots = (Path("models"), Path(registry.data_dir))
    artifacts_by_relative_path: dict[str, ModelArtifactScanResult] = {}
    for root in catalog_roots:
        resolved_root = root.resolve(strict=False)
        for artifact in scan_model_artifacts(root, force=force):
            relative_path = _catalog_relative_path(artifact.path, (resolved_root,))
            if relative_path is None:
                continue
            artifacts_by_relative_path.setdefault(relative_path.as_posix(), artifact)
    artifacts = list(artifacts_by_relative_path.values())
    return {
        **_model_catalog_payload(
            registry,
            artifacts,
            catalog_roots=catalog_roots,
        ),
        "discovered_files": len(artifacts),
        "updated_files": 0,
        "cache_hits": 0,
        "force": force,
    }


@router.get("/catalog")
def get_model_catalog(request: Request, force: bool = False) -> dict[str, Any]:
    return _read_model_catalog(_registry(request), force=force)


@router.post("/catalog/register")
@serialized_model_operation
def register_catalog_model(
    request: Request,
    payload: CatalogRegisterRequest,
) -> dict[str, Any]:
    registry = _registry(request)
    try:
        engine_path = _resolve_catalog_engine_path(registry, payload.relative_path)
        existing = _find_registered_artifact_by_path(registry, engine_path)
        if existing is not None:
            project, version, artifact = existing
            return {
                "project": asdict(project),
                "version": asdict(version),
                "artifact": asdict(artifact),
                "engine_path": str(engine_path),
                "created": False,
            }

        inspection = inspect_model_artifact(engine_path, force=True)
        project_name = _safe_component(engine_path.stem, "model")
        project = _find_project_by_name(registry, project_name)
        if project is None:
            project = registry.create_project(
                name=project_name,
                description="引用服务端 models 目录中的原始 TensorRT Engine。",
                create_asset_dir=False,
            )
        version_name = f"external-{_checksum_token(inspection.sha256)}"
        version = _find_version(registry, project.id, version_name)
        if version is None:
            version = registry.create_version(
                project_id=project.id,
                version=version_name,
                source_kind="onnx",
                source_path=str(engine_path),
                classes=["target"],
                input_shape=_ENGINE_INPUT_SHAPE_PENDING,
                create_asset_dir=False,
            )
        artifact = registry.create_artifact(
            version_id=version.id,
            kind="engine",
            path=str(engine_path),
            checksum=inspection.sha256,
            status=_registry_status_from_scan(inspection.status),
            allow_external=True,
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return {
        "project": asdict(project),
        "version": asdict(version),
        "artifact": asdict(artifact),
        "engine_path": str(engine_path),
        "created": True,
    }


@router.get("/scan")
@router.post("/scan")
def scan_models(request: Request, force: bool = False) -> dict[str, Any]:
    registry = _registry(request)
    projects = registry.list_projects()
    return {
        **_read_model_catalog(registry, force=force),
        "projects": [asdict(project) for project in projects],
        "project_count": len(projects),
        "previous_project_count": len(projects),
    }


@router.get("/artifacts/{artifact_id}/deepstream/recommendation")
def recommend_deepstream_artifact(request: Request, artifact_id: int) -> dict[str, Any]:
    registry = _registry(request)
    try:
        artifact, version, project, artifact_path = _artifact_asset_context(
            registry,
            artifact_id,
        )
        if artifact.kind != "engine":
            raise RegistryValidationError(
                "DeepStream recommendation requires a TensorRT .engine artifact"
            )
        if not artifact_path.is_file():
            raise RegistryValidationError(f"artifact file does not exist: {artifact.path}")
        resolved = recommend_engine_manifest(
            request.app.state.inference,
            artifact_path=artifact_path,
            registered_classes=list(version.classes),
            registered_input_shape=version.input_shape,
        )
        defaults, default_sources, warnings = _deepstream_model_defaults(
            artifact_path,
            version.source_path,
        )
        inference_config = getattr(getattr(request.app.state, "config", None), "inference", None)
        contract = resolved.contract
        recommendation = {
            "model_id": project.name,
            "display_name": project.name,
            **defaults,
            "input_name": contract.input_name,
            "input_shape": list(contract.input_shape),
            "input_dtype": contract.input_dtype,
            "output_name": contract.output_name,
            "output_shape": list(contract.output_shape),
            "output_dtype": contract.output_dtype,
            "class_count": len(resolved.class_names),
            "confidence_threshold": float(
                getattr(inference_config, "confidence_threshold", 0.25)
            ),
            "nms_iou_threshold": float(getattr(inference_config, "nms_threshold", 0.45)),
        }
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "artifact_id": artifact.id,
        "artifact_path": artifact.path,
        "recommendation": recommendation,
        "io_tensors": [
            {
                "name": contract.input_name,
                "shape": list(contract.input_shape),
                "dtype": contract.input_dtype,
                "mode": "input",
            },
            {
                "name": contract.output_name,
                "shape": list(contract.output_shape),
                "dtype": contract.output_dtype,
                "mode": "output",
            },
        ],
        "class_names": list(resolved.class_names),
        "output_has_objectness": resolved.output_has_objectness,
        "sources": {
            "input_contract": "tensorrt_engine_probe",
            "output_contract": "tensorrt_engine_probe",
            "class_contract": "engine_output_with_registry_and_filename_hints",
            **default_sources,
        },
        "warnings": warnings,
    }


@router.post("/examples/yolov8n/prepare")
def prepare_yolov8n_example(request: Request) -> dict[str, Any]:
    registry = _registry(request)
    project = _find_project_by_name(registry, "yolov8n")
    if project is None:
        project = registry.create_project(
            name="yolov8n",
            description="Ultralytics YOLOv8n 开源测试模型，用于图片输入源和推理链路验证。",
        )
    version = _find_version(registry, project.id, "v8n")
    if version is None:
        version = registry.create_version(
            project_id=project.id,
            version="v8n",
            source_kind="pt",
            source_path=YOLOV8N_URL,
            classes=YOLOV8N_CLASSES,
            input_shape="1x3x640x640",
        )
    model_path = Path(registry.data_dir) / project.name / version.version / "yolov8n.pt"
    downloaded = False
    if not model_path.exists():
        _download_file(YOLOV8N_URL, model_path)
        downloaded = True
    checksum = _sha256(model_path)
    artifact = _find_artifact(registry, version.id, "yolov8n.pt")
    if artifact is None:
        artifact = registry.create_artifact(
            version_id=version.id,
            kind="pt",
            path="yolov8n.pt",
            checksum=checksum,
            status="ready",
        )
    return {
        "project": asdict(project),
        "version": asdict(version),
        "artifact": asdict(artifact),
        "downloaded": downloaded,
        "url": YOLOV8N_URL,
    }


@router.post("/upload")
async def upload_model(
    request: Request,
    project_name: str = Form(...),
    version: str = Form(...),
    description: str = Form(""),
    classes: str = Form("target"),
    input_shape: str = Form(""),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    registry = _registry(request)
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="model filename is required")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="model file is empty")
    tmp_path: Path | None = None
    try:
        kind = _artifact_kind_from_filename(filename)
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as handle:
            handle.write(content)
            tmp_path = Path(handle.name)
        if kind == "engine":
            resolved_input_shape = _ENGINE_INPUT_SHAPE_PENDING
        elif input_shape.strip():
            resolved_input_shape = _normalize_input_shape(input_shape)
        else:
            resolved_input_shape = _detect_input_shape(tmp_path, {})
        checksum = _sha256(tmp_path)
        project = _find_project_by_name(registry, project_name)
        if project is None:
            project = registry.create_project(
                name=project_name,
                description=description,
            )
        model_version, artifact = _select_artifact_version(
            registry,
            project=project,
            requested_version_name=version,
            kind=kind,
            source_path=filename,
            classes=_parse_classes(classes),
            input_shape=resolved_input_shape,
            checksum=checksum,
        )
        asset_filename = (
            artifact.path
            if artifact is not None
            else _immutable_artifact_filename(filename, checksum)
        )
        asset_path = (
            Path(registry.data_dir)
            / project.name
            / model_version.version
            / asset_filename
        )
        if artifact is None or not asset_path.is_file():
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(content)
        artifact_status = _artifact_registry_status(kind, tmp_path)
        if artifact is None:
            artifact = registry.create_artifact(
                version_id=model_version.id,
                kind=kind,
                path=asset_filename,
                checksum=checksum,
                status=artifact_status,
            )
        elif (
            artifact.status != artifact_status
            and not (artifact.status == "ready" and artifact_status == "pending")
        ):
            artifact = registry.update_artifact_status(
                artifact.id,
                artifact_status,
            )
    except RegistryError as exc:
        logger.warning("model upload rejected filename=%s error=%s", filename, exc)
        raise _as_http_error(exc) from exc
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return {
        "project": asdict(project),
        "version": asdict(model_version),
        "artifact": asdict(artifact),
    }


@router.post("/projects")
def create_project(
    request: Request,
    payload: ProjectCreateRequest,
) -> dict[str, Any]:
    try:
        project = _registry(request).create_project(
            name=payload.name,
            description=payload.description,
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return asdict(project)


@router.get("/projects/{project_id}/versions")
def list_versions(request: Request, project_id: int) -> list[dict[str, Any]]:
    registry = _registry(request)
    try:
        _require_project(registry, project_id)
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return [asdict(version) for version in registry.list_versions(project_id)]


@router.post("/projects/{project_id}/versions")
def create_version(
    request: Request,
    project_id: int,
    payload: VersionCreateRequest,
) -> dict[str, Any]:
    try:
        version = _registry(request).create_version(
            project_id=project_id,
            version=payload.version,
            source_kind=payload.source_kind,
            source_path=payload.source_path,
            classes=payload.classes,
            input_shape=payload.input_shape,
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return asdict(version)


@router.get("/versions/{version_id}/artifacts")
def list_artifacts(request: Request, version_id: int) -> list[dict[str, Any]]:
    registry = _registry(request)
    version = registry.get_version(version_id)
    if version is None:
        raise _as_http_error(RegistryNotFoundError(f"unknown version id: {version_id}"))
    project = registry.get_project(version.project_id)
    if project is None:
        raise _as_http_error(RegistryNotFoundError(f"unknown project id: {version.project_id}"))
    return [
        {
            **asdict(artifact),
            "size_bytes": _file_size_bytes(registry.resolve_artifact_path(artifact)),
        }
        for artifact in registry.list_artifacts(version_id)
    ]


def _file_size_bytes(path: Path) -> int | None:
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


@router.post("/versions/{version_id}/artifacts")
def create_artifact(
    request: Request,
    version_id: int,
    payload: ArtifactCreateRequest,
) -> dict[str, Any]:
    registry = _registry(request)
    try:
        artifact = registry.create_artifact(
            version_id=version_id,
            kind=payload.kind,
            path=payload.path,
            checksum=payload.checksum,
            status=payload.status,
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return asdict(artifact)


@router.get("/jobs")
def list_conversion_jobs(
    request: Request,
    version_id: int | None = None,
) -> list[dict[str, Any]]:
    return _list_conversion_jobs(request, version_id=version_id)


@router.post("/jobs/list")
def post_list_conversion_jobs(
    request: Request,
    payload: ConversionJobListRequest,
) -> list[dict[str, Any]]:
    return _list_conversion_jobs(request, version_id=payload.version_id)


def _list_conversion_jobs(
    request: Request,
    *,
    version_id: int | None,
) -> list[dict[str, Any]]:
    registry = _registry(request)
    if version_id is not None:
        try:
            _require_version(registry, version_id)
        except RegistryError as exc:
            raise _as_http_error(exc) from exc
    return [asdict(job) for job in registry.list_conversion_jobs(version_id=version_id)]


@router.post("/versions/{version_id}/jobs")
def create_conversion_job(
    request: Request,
    version_id: int,
    payload: ConversionJobCreateRequest,
) -> dict[str, Any]:
    try:
        job = _registry(request).create_conversion_job(
            version_id=version_id,
            target_kind=payload.target_kind,
            command=payload.command,
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return asdict(job)


@router.get("/jobs/{job_id}")
def get_conversion_job(request: Request, job_id: int) -> dict[str, Any]:
    job = _registry(request).get_conversion_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown conversion job id: {job_id}",
        )
    return asdict(job)


@router.post("/jobs/{job_id}/finish")
def finish_conversion_job(
    request: Request,
    job_id: int,
    payload: ConversionJobFinishRequest,
) -> dict[str, bool]:
    try:
        _registry(request).finish_conversion_job(
            job_id=job_id,
            status=payload.status,
            log=payload.log,
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return {"ok": True}


@router.post("/projects/{project_id}/publish")
@serialized_model_operation
def publish(
    request: Request,
    project_id: int,
    payload: PublishRequest,
) -> dict[str, Any]:
    registry = _registry(request)
    paused_for_switch = False
    try:
        _require_project(registry, project_id)
        _require_control_inference_policy(request)
        artifact_path, classes, input_shape = _resolve_runnable_artifact(
            registry,
            project_id=project_id,
            artifact_id=payload.artifact_id,
        )
        candidate, candidate_status = _prepare_runnable_artifact(
            request,
            artifact_path=artifact_path,
            classes=classes,
            input_shape=input_shape,
        )
        candidate_classes = candidate_status.get("classes")
        if isinstance(candidate_classes, list) and candidate_classes:
            classes = [str(item) for item in candidate_classes]
        input_shape = _actual_runtime_input_shape(candidate_status, input_shape)
        get_artifact = getattr(registry, "get_artifact", None)
        artifact = get_artifact(payload.artifact_id) if callable(get_artifact) else None
        if artifact is not None and artifact.status == "pending":
            registry.update_artifact_status(payload.artifact_id, "ready")
        paused_for_switch = _pause_runtime_pipeline_for_model_switch(request)
        try:
            deployment = registry.publish(
                project_id=project_id,
                artifact_id=payload.artifact_id,
            )
        except RegistryError:
            _close_candidate(candidate)
            raise
        _sync_artifact_version_input_shape(
            registry,
            artifact_id=payload.artifact_id,
            input_shape=input_shape,
        )
        if candidate is None:
            request.app.state.inference.unload(
                "TensorRT engine ownership delegated to DeepStream nvinfer"
            )
            _clear_runtime_pipeline_after_model_switch(
                request,
                "DeepStream active model changed",
            )
        else:
            request.app.state.inference.commit(
                candidate,
                artifact_path=artifact_path,
                classes=classes,
                input_shape=input_shape,
            )
        set_profile_activation(artifact_path, active=True)
        _set_previous_profile_inactive(
            registry,
            deployment.previous_artifact_id,
            deployment.artifact_id,
        )
    except RegistryError as exc:
        _resume_runtime_pipeline_after_model_switch(request, paused_for_switch)
        record_switch_error = getattr(request.app.state.inference, "record_switch_error", None)
        if callable(record_switch_error):
            record_switch_error(str(exc))
        logger.warning(
            "model publish rejected project_id=%s artifact_id=%s error=%s",
            project_id,
            payload.artifact_id,
            exc,
        )
        raise _as_http_error(exc) from exc
    except Exception:
        _resume_runtime_pipeline_after_model_switch(request, paused_for_switch)
        raise
    _resume_runtime_pipeline_after_model_switch(request, paused_for_switch)
    inference_status = _inference_status(request)
    return {
        "deployment": asdict(deployment),
        "inference": inference_status,
        "report": _model_switch_report(
            action="publish",
            deployment=deployment,
            artifact_path=artifact_path,
            classes=classes,
            input_shape=input_shape,
            inference_status=inference_status,
        ),
    }


@router.post("/projects/{project_id}/rollback")
@serialized_model_operation
def rollback(request: Request, project_id: int) -> dict[str, Any]:
    registry = _registry(request)
    paused_for_switch = False
    try:
        _require_project(registry, project_id)
        _require_control_inference_policy(request)
        current = registry.get_deployment(project_id)
        paused_for_switch = _pause_runtime_pipeline_for_model_switch(request)
        if current is None or current.previous_artifact_id is None:
            deployment = registry.rollback(project_id=project_id)
            artifact_path, classes, input_shape = _resolve_runnable_artifact(
                registry,
                project_id=project_id,
                artifact_id=deployment.artifact_id,
            )
            _load_published_artifact(request, registry, deployment.artifact_id)
            set_profile_activation(artifact_path, active=True)
            _set_previous_profile_inactive(
                registry,
                deployment.previous_artifact_id,
                deployment.artifact_id,
            )
            _resume_runtime_pipeline_after_model_switch(request, paused_for_switch)
            inference_status = _inference_status(request)
            input_shape = _actual_runtime_input_shape(inference_status, input_shape)
            _sync_artifact_version_input_shape(
                registry,
                artifact_id=deployment.artifact_id,
                input_shape=input_shape,
            )
            return {
                "deployment": asdict(deployment),
                "inference": inference_status,
                "report": _model_switch_report(
                    action="rollback",
                    deployment=deployment,
                    artifact_path=artifact_path,
                    classes=classes,
                    input_shape=input_shape,
                    inference_status=inference_status,
                ),
            }
        artifact_path, classes, input_shape = _resolve_runnable_artifact(
            registry,
            project_id=project_id,
            artifact_id=current.previous_artifact_id,
        )
        candidate, candidate_status = _prepare_runnable_artifact(
            request,
            artifact_path=artifact_path,
            classes=classes,
            input_shape=input_shape,
        )
        input_shape = _actual_runtime_input_shape(candidate_status, input_shape)
        try:
            deployment = registry.rollback(project_id=project_id)
        except RegistryError:
            _close_candidate(candidate)
            raise
        _sync_artifact_version_input_shape(
            registry,
            artifact_id=current.previous_artifact_id,
            input_shape=input_shape,
        )
        if candidate is None:
            request.app.state.inference.unload(
                "TensorRT engine ownership delegated to DeepStream nvinfer"
            )
            _clear_runtime_pipeline_after_model_switch(
                request,
                "DeepStream rollback changed active model",
            )
        else:
            request.app.state.inference.commit(
                candidate,
                artifact_path=artifact_path,
                classes=classes,
                input_shape=input_shape,
            )
        set_profile_activation(artifact_path, active=True)
        _set_previous_profile_inactive(
            registry,
            deployment.previous_artifact_id,
            deployment.artifact_id,
        )
    except RegistryError as exc:
        _resume_runtime_pipeline_after_model_switch(request, paused_for_switch)
        record_switch_error = getattr(request.app.state.inference, "record_switch_error", None)
        if callable(record_switch_error):
            record_switch_error(str(exc))
        logger.warning("model rollback rejected project_id=%s error=%s", project_id, exc)
        raise _as_http_error(exc) from exc
    except Exception:
        _resume_runtime_pipeline_after_model_switch(request, paused_for_switch)
        raise
    _resume_runtime_pipeline_after_model_switch(request, paused_for_switch)
    inference_status = _inference_status(request)
    return {
        "deployment": asdict(deployment),
        "inference": inference_status,
        "report": _model_switch_report(
            action="rollback",
            deployment=deployment,
            artifact_path=artifact_path,
            classes=classes,
            input_shape=input_shape,
            inference_status=inference_status,
        ),
    }
