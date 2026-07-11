from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import tempfile
from dataclasses import asdict, dataclass
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
    TensorSpec,
    build_engine_manifest,
    inspect_model_artifact,
    scan_model_artifacts,
    write_manifest,
)
from novasight.inference import parse_tensor_input_shape
from novasight.inference.input import normalize_tensor_dtype
from novasight.model_registry.manifest import read_manifest, validate_manifest_engine_artifact

router = APIRouter(prefix="/api/models")
logger = logging.getLogger("novasight.api.models")
_MODEL_SYNC_LOCK = threading.RLock()
_MODEL_SYNC_CACHE: dict[tuple[str, str], tuple[int, ...]] = {}


@dataclass(frozen=True)
class ModelDirectorySyncResult:
    discovered_files: int
    updated_files: int
    cache_hits: int


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


class DeepStreamPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: StrictStr
    display_name: StrictStr
    runtime_precision: StrictStr = "fp16"
    input_name: StrictStr = "images"
    input_shape: list[StrictInt]
    input_dtype: StrictStr = "float32"
    input_color_format: StrictStr = "RGB"
    input_scale_factor: float = 1.0 / 255.0
    maintain_aspect_ratio: bool = False
    symmetric_padding: bool = False
    output_name: StrictStr = "output0"
    output_shape: list[StrictInt]
    output_dtype: StrictStr = "float32"
    class_count: StrictInt
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45


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
    return f"{path.stem}.{_checksum_token(checksum)}{path.suffix.lower()}"


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


def _classes_from_sidecar(sidecar: dict[str, Any]) -> list[str]:
    classes_value = sidecar.get("classes", sidecar.get("names", ["target"]))
    if isinstance(classes_value, dict):
        ordered = sorted(
            ((int(key), value) for key, value in classes_value.items() if str(key).isdigit()),
            key=lambda item: item[0],
        )
        classes_value = [value for _key, value in ordered]
    if not isinstance(classes_value, list):
        return ["target"]
    return [str(item) for item in classes_value if str(item).strip()] or ["target"]


def _sync_models_directory(
    registry: ModelRegistry,
    *,
    force: bool = False,
) -> ModelDirectorySyncResult:
    with _MODEL_SYNC_LOCK:
        roots = [Path(registry.data_dir), Path("models")]
        seen: set[Path] = set()
        seen_cache_keys: set[tuple[str, str]] = set()
        updated_files = 0
        cache_hits = 0
        registry_key = str(Path(registry.db_path).resolve(strict=False))
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            root = root.resolve(strict=False)
            for model_file in sorted(root.rglob("*")):
                if not model_file.is_file() or model_file.suffix.lower() not in {".onnx", ".engine"}:
                    continue
                resolved = model_file.resolve(strict=False)
                if resolved in seen:
                    continue
                seen.add(resolved)
                cache_key = (registry_key, str(resolved))
                seen_cache_keys.add(cache_key)
                signature = _model_file_signature(resolved)
                if not force and _MODEL_SYNC_CACHE.get(cache_key) == signature:
                    cache_hits += 1
                    continue
                if _sync_model_file(registry, root, resolved, force=force):
                    _MODEL_SYNC_CACHE[cache_key] = signature
                    updated_files += 1
        stale_keys = [
            key
            for key in _MODEL_SYNC_CACHE
            if key[0] == registry_key and key not in seen_cache_keys
        ]
        for key in stale_keys:
            _MODEL_SYNC_CACHE.pop(key, None)
        return ModelDirectorySyncResult(
            discovered_files=len(seen),
            updated_files=updated_files,
            cache_hits=cache_hits,
        )


def _sync_model_file(
    registry: ModelRegistry,
    root: Path,
    model_file: Path,
    *,
    force: bool = False,
) -> bool:
    try:
        kind = _artifact_kind_from_filename(model_file.name)
        relative = model_file.relative_to(root)
        if len(relative.parts) >= 3:
            project_name = _safe_component(relative.parts[0], model_file.stem)
            version_name = _safe_component(relative.parts[1], "default")
            filename = Path(*relative.parts[2:]).name
        else:
            project_name = _safe_component(model_file.stem, "model")
            version_name = "default"
            filename = model_file.name
        sidecar = _model_sidecar(model_file)
        classes = _classes_from_sidecar(sidecar)
        input_shape = _detect_input_shape(model_file, sidecar)
        existing_project = _find_project_by_name(registry, project_name)
        existing_version = (
            _find_version(registry, existing_project.id, version_name)
            if existing_project is not None
            else None
        )
        if existing_version is not None:
            if "classes" not in sidecar and "names" not in sidecar:
                classes = list(existing_version.classes)
            if (
                kind == "engine"
                and "input_shape" not in sidecar
                and _detect_input_shape_from_filename(model_file) is None
            ):
                input_shape = existing_version.input_shape
        inspection = inspect_model_artifact(model_file, force=force)
        checksum = inspection.sha256
        project = existing_project
        if project is None:
            project = registry.create_project(
                name=project_name,
                description="从服务端 models 目录自动发现的模型。",
            )
        requested_version = _find_version(registry, project.id, version_name)
        managed_artifact = (
            _find_artifact(registry, requested_version.id, filename)
            if requested_version is not None
            else None
        )
        if (
            managed_artifact is not None
            and managed_artifact.checksum == checksum
            and model_file.resolve(strict=False).is_relative_to(
                Path(registry.data_dir).resolve(strict=False)
            )
        ):
            artifact_status = (
                _registry_status_from_scan(inspection.status)
                if kind == "engine"
                else "ready"
            )
            if managed_artifact.status != artifact_status:
                registry.update_artifact_status(
                    managed_artifact.id,
                    artifact_status,
                )
            return True
        version, existing_artifact = _select_artifact_version(
            registry,
            project=project,
            requested_version_name=version_name,
            kind=kind,
            source_path=model_file.as_posix(),
            classes=classes,
            input_shape=input_shape,
            checksum=checksum,
        )
        asset_filename = (
            existing_artifact.path
            if existing_artifact is not None
            else _immutable_artifact_filename(filename, checksum)
        )
        asset_path = Path(registry.data_dir) / project.name / version.version / asset_filename
        if (
            model_file.resolve(strict=False) != asset_path.resolve(strict=False)
            and (existing_artifact is None or not asset_path.is_file())
        ):
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_file, asset_path)
        artifact_status = (
            _registry_status_from_scan(inspection.status)
            if kind == "engine"
            else "ready"
        )
        if existing_artifact is None:
            registry.create_artifact(
                version_id=version.id,
                kind=kind,
                path=asset_filename,
                checksum=checksum,
                status=artifact_status,
            )
        elif existing_artifact.status != artifact_status:
            registry.update_artifact_status(
                existing_artifact.id,
                artifact_status,
            )
        asset_cache_key = (
            str(Path(registry.db_path).resolve(strict=False)),
            str(asset_path.resolve(strict=False)),
        )
        _MODEL_SYNC_CACHE[asset_cache_key] = _model_file_signature(asset_path)
        return True
    except (OSError, RegistryError) as exc:
        logger.warning("model directory sync skipped path=%s error=%s", model_file, exc)
        return False


def _model_file_signature(path: Path) -> tuple[int, ...]:
    return (
        *_path_stat_signature(path),
        *_path_stat_signature(path.with_suffix(".json")),
        *_path_stat_signature(path.with_name("model.manifest.json")),
    )


def _path_stat_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (-1, -1)
    return (int(stat.st_size), int(stat.st_mtime_ns))


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
    artifact_path = Path(registry.data_dir) / project.name / version.version / artifact.path
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
    artifact_path = (
        Path(registry.data_dir) / project.name / version.version / artifact.path
    )
    if _deepstream_selected(request):
        _validate_deepstream_artifact(artifact_path)
        request.app.state.inference.unload(
            "TensorRT engine ownership delegated to DeepStream nvinfer"
        )
        _clear_runtime_pipeline_after_model_switch(
            request,
            "DeepStream active model changed",
        )
        return
    request.app.state.inference.load(
        artifact_path,
        list(version.classes),
        version.input_shape,
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
        Path(registry.data_dir) / project.name / version.version / artifact.path,
        list(version.classes),
        version.input_shape,
    )


def _probe_runnable_artifact(
    request: Request,
    *,
    artifact_path: Path,
    classes: list[str],
    input_shape: str,
) -> dict[str, Any]:
    inference = request.app.state.inference
    probe = getattr(inference, "probe", None)
    if not callable(probe):
        raise RegistryValidationError("inference runtime does not support safe model switching")
    status = dict(probe(artifact_path, classes, input_shape))
    if status.get("loaded") is not True:
        reason = status.get("reason") or "model probe failed"
        raise RegistryValidationError(f"model switch rejected: {reason}")
    return status


def _prepare_runnable_artifact(
    request: Request,
    *,
    artifact_path: Path,
    classes: list[str],
    input_shape: str,
) -> tuple[Any, dict[str, Any]]:
    if _deepstream_selected(request):
        manifest = _validate_deepstream_artifact(artifact_path)
        return None, {
            "selected": "deepstream_nvinfer",
            "available": True,
            "loaded": True,
            "input_shape": "x".join(str(value) for value in manifest.input.shape),
            "model_fingerprint": manifest.model_fingerprint,
            "reason": "validated for pipeline-owned nvinfer loading",
        }
    inference = request.app.state.inference
    prepare = getattr(inference, "prepare", None)
    if not callable(prepare):
        raise RegistryValidationError("inference runtime does not support safe model switching")
    try:
        candidate, status = prepare(artifact_path, classes, input_shape)
        if status.get("loaded") is not True:
            _close_candidate(candidate)
            reason = status.get("reason") or "candidate runtime did not report loaded=true"
            raise RegistryValidationError(f"model switch rejected: {reason}")
        return candidate, status
    except RegistryValidationError:
        raise
    except Exception as exc:
        reason = f"model switch rejected: {exc}"
        record_switch_error = getattr(inference, "record_switch_error", None)
        if callable(record_switch_error):
            record_switch_error(reason)
        logger.exception("model switch prepare failed artifact=%s", artifact_path)
        raise RegistryValidationError(reason) from exc


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
    if pipeline is None or getattr(pipeline, "running", False) is not True:
        return False
    logger.info("pausing runtime pipeline for model switch")
    pipeline.stop()
    return True


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


def _validate_deepstream_prepare_payload(payload: DeepStreamPrepareRequest) -> bool:
    input_shape = [int(item) for item in payload.input_shape]
    output_shape = [int(item) for item in payload.output_shape]
    class_count = int(payload.class_count)
    precision = payload.runtime_precision.strip().lower()
    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(
            "DeepStream runtime_precision must be fp32, fp16, or int8 "
            f"(got {payload.runtime_precision})"
        )
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError(
            f"DeepStream input_shape must be [1, 3, H, W], got {input_shape}"
        )
    if len(output_shape) != 3 or output_shape[0] != 1:
        raise ValueError(
            "DeepStream YOLO output_shape must be [1, channels, candidates] "
            f"or [1, candidates, channels], got {output_shape}"
        )
    if class_count <= 0:
        raise ValueError(f"DeepStream class_count must be positive, got {class_count}")
    color_format = payload.input_color_format.strip().upper()
    if color_format not in {"RGB", "BGR", "GRAY", "GREY"}:
        raise ValueError(
            "DeepStream input_color_format must be RGB, BGR, or GRAY "
            f"(got {payload.input_color_format})"
        )
    if float(payload.input_scale_factor) <= 0.0:
        raise ValueError(
            "DeepStream input_scale_factor must be positive "
            f"(got {payload.input_scale_factor})"
        )
    channel_dimensions = {4 + class_count, 5 + class_count}
    matching_channels = [
        value for value in output_shape[1:] if int(value) in channel_dimensions
    ]
    if len(matching_channels) != 1:
        raise ValueError(
            "DeepStream YOLO output must have exactly one dimension equal to "
            "4 + class_count or 5 + class_count "
            f"(shape={output_shape}, class_count={class_count})"
        )
    return int(matching_channels[0]) == 5 + class_count


def _probe_deepstream_engine_contract(
    request: Request,
    *,
    artifact_path: Path,
    classes: list[str],
    registered_input_shape: str,
) -> dict[str, Any]:
    inference = getattr(request.app.state, "inference", None)
    probe = getattr(inference, "probe", None)
    if not callable(probe):
        raise RegistryValidationError(
            "TensorRT engine probe is unavailable; refusing to guess DeepStream tensor bindings"
        )
    status = dict(probe(artifact_path, classes, registered_input_shape))
    if status.get("loaded") is not True:
        raise RegistryValidationError(
            f"TensorRT engine probe failed: {status.get('reason') or 'loaded=false'}"
        )
    input_shape = _parse_runtime_shape(status.get("input_shape"), "input_shape")
    output_shape = _parse_runtime_shape(status.get("output_shape"), "output_shape")
    input_name = str(status.get("input_name") or "").strip()
    output_name = str(status.get("output_name") or "").strip()
    input_dtype = normalize_tensor_dtype(status.get("input_dtype"))
    output_dtype = normalize_tensor_dtype(status.get("output_dtype"))
    if not input_name or not output_name:
        raise RegistryValidationError(
            "TensorRT engine probe did not expose input/output tensor names"
        )
    return {
        "input_name": input_name,
        "input_shape": input_shape,
        "input_dtype": input_dtype,
        "output_name": output_name,
        "output_shape": output_shape,
        "output_dtype": output_dtype,
    }


def _parse_runtime_shape(value: object, label: str) -> list[int]:
    parts = [item for item in re.split(r"[xX,\s]+", str(value or "").strip()) if item]
    try:
        shape = [int(item) for item in parts]
    except ValueError as exc:
        raise RegistryValidationError(
            f"TensorRT engine probe returned invalid {label}: {value}"
        ) from exc
    if not shape or any(item <= 0 for item in shape):
        raise RegistryValidationError(
            f"TensorRT engine probe returned unresolved {label}: {value}"
        )
    return shape


def _validate_deepstream_artifact(artifact_path: Path):
    if artifact_path.suffix.lower() != ".engine":
        raise RegistryValidationError("deepstream_nvinfer requires a TensorRT .engine artifact")
    manifest_path = artifact_path.with_name("model.manifest.json")
    if not manifest_path.is_file():
        raise RegistryValidationError(f"model manifest missing: {manifest_path}")
    try:
        manifest = read_manifest(manifest_path)
        validate_manifest_engine_artifact(manifest, artifact_path)
    except ValueError as exc:
        raise RegistryValidationError(str(exc)) from exc
    return manifest


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


@router.get("/scan")
@router.post("/scan")
def scan_models(request: Request, force: bool = False) -> dict[str, Any]:
    registry = _registry(request)
    before = len(registry.list_projects())
    sync_result = _sync_models_directory(registry, force=force)
    projects = registry.list_projects()
    artifacts = scan_model_artifacts(Path(registry.data_dir), force=force)
    return {
        "projects": [asdict(project) for project in projects],
        "project_count": len(projects),
        "previous_project_count": before,
        "discovered_files": sync_result.discovered_files,
        "updated_files": sync_result.updated_files,
        "cache_hits": sync_result.cache_hits,
        "force": force,
        "artifacts": [
            _artifact_scan_payload(registry, artifact)
            for artifact in artifacts
        ],
        "artifact_status_counts": _artifact_scan_counts(artifacts),
    }


@router.post("/artifacts/{artifact_id}/deepstream/prepare")
def prepare_deepstream_artifact(
    request: Request,
    artifact_id: int,
    payload: DeepStreamPrepareRequest,
) -> dict[str, Any]:
    registry = _registry(request)
    try:
        artifact, version, _project, artifact_path = _artifact_asset_context(
            registry,
            artifact_id,
        )
        if artifact.kind != "engine":
            raise RegistryValidationError(
                "DeepStream prepare requires a TensorRT .engine artifact"
            )
        if not artifact_path.is_file():
            raise RegistryValidationError(f"artifact file does not exist: {artifact.path}")
        class_names = list(version.classes)
        if int(payload.class_count) != len(class_names):
            raise RegistryValidationError(
                "DeepStream class_count must match the model version classes "
                f"(class_count={payload.class_count}, classes={len(class_names)})"
            )
        engine_contract = _probe_deepstream_engine_contract(
            request,
            artifact_path=artifact_path,
            classes=class_names,
            registered_input_shape=version.input_shape,
        )
        requested_input_shape = [int(item) for item in payload.input_shape]
        requested_output_shape = [int(item) for item in payload.output_shape]
        if requested_input_shape != engine_contract["input_shape"]:
            raise RegistryValidationError(
                "confirmed DeepStream input shape does not match TensorRT engine "
                f"(confirmed={requested_input_shape}, engine={engine_contract['input_shape']})"
            )
        if requested_output_shape != engine_contract["output_shape"]:
            raise RegistryValidationError(
                "confirmed DeepStream output shape does not match TensorRT engine "
                f"(confirmed={requested_output_shape}, engine={engine_contract['output_shape']})"
            )
        output_has_objectness = _validate_deepstream_prepare_payload(payload)
        manifest = build_engine_manifest(
            model_id=payload.model_id,
            display_name=payload.display_name,
            engine_path=artifact_path,
            input_spec=TensorSpec(
                name=engine_contract["input_name"],
                shape=engine_contract["input_shape"],
                dtype=engine_contract["input_dtype"],
                layout="NCHW",
            ),
            output_spec=TensorSpec(
                name=engine_contract["output_name"],
                shape=engine_contract["output_shape"],
                dtype=engine_contract["output_dtype"],
                layout="NCHW",
            ),
            class_count=int(payload.class_count),
            class_names=class_names,
            confidence_threshold=float(payload.confidence_threshold),
            nms_iou_threshold=float(payload.nms_iou_threshold),
            runtime_precision=payload.runtime_precision,
            input_color_format=payload.input_color_format,
            input_scale_factor=float(payload.input_scale_factor),
            maintain_aspect_ratio=bool(payload.maintain_aspect_ratio),
            symmetric_padding=bool(payload.symmetric_padding),
            output_has_objectness=output_has_objectness,
            validated=True,
        )
        manifest_path = artifact_path.with_name("model.manifest.json")
        write_manifest(manifest, manifest_path)
        result = inspect_model_artifact(artifact_path, force=True)
        if result.status != "ready":
            raise RegistryValidationError(
                f"generated DeepStream manifest did not validate: {result.reason}"
            )
        updated_artifact = registry.update_artifact_status(
            artifact.id,
            "ready",
            checksum=result.sha256,
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": result.status,
        "reason": result.reason,
        "artifact": asdict(updated_artifact),
        "manifest_path": _relative_registry_path(registry, manifest_path),
        "model_fingerprint": result.model_fingerprint,
        "nvinfer_config_owner": "runtime",
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
        resolved_input_shape = (
            _normalize_input_shape(input_shape)
            if input_shape.strip()
            else _detect_input_shape(tmp_path, {})
        )
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
    asset_dir = Path(registry.data_dir) / project.name / version.version
    return [
        {
            **asdict(artifact),
            "size_bytes": _file_size_bytes(asset_dir / artifact.path),
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
def publish(
    request: Request,
    project_id: int,
    payload: PublishRequest,
) -> dict[str, Any]:
    registry = _registry(request)
    paused_for_switch = False
    try:
        _require_project(registry, project_id)
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
def rollback(request: Request, project_id: int) -> dict[str, Any]:
    registry = _registry(request)
    paused_for_switch = False
    try:
        _require_project(registry, project_id)
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
