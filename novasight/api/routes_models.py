from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from novasight.model_registry import (
    ModelRegistry,
    RegistryConflictError,
    RegistryError,
    RegistryNotFoundError,
    RegistryValidationError,
)
from novasight.inference import parse_tensor_input_shape

router = APIRouter(prefix="/api/models")
logger = logging.getLogger("novasight.api.models")

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


def _sync_models_directory(registry: ModelRegistry) -> None:
    roots = [Path(registry.data_dir), Path("models")]
    seen: set[Path] = set()
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
            _sync_model_file(registry, root, resolved)


def _sync_model_file(registry: ModelRegistry, root: Path, model_file: Path) -> None:
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
        project = _find_project_by_name(registry, project_name)
        if project is None:
            project = registry.create_project(
                name=project_name,
                description="从服务端 models 目录自动发现的模型。",
            )
        version = _find_version(registry, project.id, version_name)
        if version is None:
            version = registry.create_version(
                project_id=project.id,
                version=version_name,
                source_kind=_source_kind_from_artifact(kind),
                source_path=model_file.as_posix(),
                classes=classes,
                input_shape=input_shape,
            )
        elif version.input_shape != input_shape:
            version = registry.update_version_input_shape(version.id, input_shape)
        asset_path = Path(registry.data_dir) / project.name / version.version / filename
        if model_file.resolve(strict=False) != asset_path.resolve(strict=False):
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_file, asset_path)
        checksum = _sha256(asset_path)
        if _find_artifact(registry, version.id, filename) is None:
            registry.create_artifact(
                version_id=version.id,
                kind=kind,
                path=filename,
                checksum=checksum,
                status="ready",
            )
    except (OSError, RegistryError) as exc:
        logger.warning("model directory sync skipped path=%s error=%s", model_file, exc)
        return


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
        request.app.state.inference.disable(
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
    if artifact.status != "ready":
        raise RegistryValidationError("only ready artifacts can be published")
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
    inference = request.app.state.inference
    prepare = getattr(inference, "prepare", None)
    if not callable(prepare):
        raise RegistryValidationError("inference runtime does not support safe model switching")
    try:
        return prepare(artifact_path, classes, input_shape)
    except Exception as exc:
        raise RegistryValidationError(f"model switch rejected: {exc}") from exc


def _close_candidate(candidate: Any) -> None:
    close = getattr(candidate, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _inference_status(request: Request) -> dict[str, Any]:
    inference = getattr(request.app.state, "inference", None)
    status = getattr(inference, "status", None)
    if not callable(status):
        return {"available": False, "loaded": False, "reason": "inference runtime unavailable"}
    return dict(status())


@router.get("/projects")
def list_projects(request: Request) -> list[dict[str, Any]]:
    registry = _registry(request)
    _sync_models_directory(registry)
    return [asdict(project) for project in registry.list_projects()]


@router.post("/scan")
def scan_models(request: Request) -> dict[str, Any]:
    registry = _registry(request)
    before = len(registry.list_projects())
    _sync_models_directory(registry)
    projects = registry.list_projects()
    return {"projects": [asdict(project) for project in projects], "project_count": len(projects), "previous_project_count": before}


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
        project = _find_project_by_name(registry, project_name)
        if project is None:
            project = registry.create_project(
                name=project_name,
                description=description,
            )
        model_version = _find_version(registry, project.id, version)
        if model_version is None:
            model_version = registry.create_version(
                project_id=project.id,
                version=version,
                source_kind=_source_kind_from_artifact(kind),
                source_path=filename,
                classes=_parse_classes(classes),
                input_shape=resolved_input_shape,
            )
        elif model_version.input_shape != resolved_input_shape:
            model_version = registry.update_version_input_shape(
                model_version.id,
                resolved_input_shape,
            )
        asset_path = Path(registry.data_dir) / project.name / model_version.version / filename
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(content)
        checksum = _sha256(asset_path)
        artifact = _find_artifact(registry, model_version.id, filename)
        if artifact is None:
            artifact = registry.create_artifact(
                version_id=model_version.id,
                kind=kind,
                path=filename,
                checksum=checksum,
                status="ready",
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
    try:
        _require_version(registry, version_id)
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return [asdict(artifact) for artifact in registry.list_artifacts(version_id)]


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
    try:
        _require_project(registry, project_id)
        artifact_path, classes, input_shape = _resolve_runnable_artifact(
            registry,
            project_id=project_id,
            artifact_id=payload.artifact_id,
        )
        candidate, _candidate_status = _prepare_runnable_artifact(
            request,
            artifact_path=artifact_path,
            classes=classes,
            input_shape=input_shape,
        )
        try:
            deployment = registry.publish(
                project_id=project_id,
                artifact_id=payload.artifact_id,
            )
        except RegistryError:
            _close_candidate(candidate)
            raise
        request.app.state.inference.commit(
            candidate,
            artifact_path=artifact_path,
            classes=classes,
            input_shape=input_shape,
        )
    except RegistryError as exc:
        logger.warning(
            "model publish rejected project_id=%s artifact_id=%s error=%s",
            project_id,
            payload.artifact_id,
            exc,
        )
        raise _as_http_error(exc) from exc
    return {
        "deployment": asdict(deployment),
        "inference": _inference_status(request),
    }


@router.post("/projects/{project_id}/rollback")
def rollback(request: Request, project_id: int) -> dict[str, Any]:
    registry = _registry(request)
    try:
        _require_project(registry, project_id)
        current = registry.get_deployment(project_id)
        if current is None or current.previous_artifact_id is None:
            deployment = registry.rollback(project_id=project_id)
            _load_published_artifact(request, registry, deployment.artifact_id)
            return {
                "deployment": asdict(deployment),
                "inference": _inference_status(request),
            }
        artifact_path, classes, input_shape = _resolve_runnable_artifact(
            registry,
            project_id=project_id,
            artifact_id=current.previous_artifact_id,
        )
        candidate, _candidate_status = _prepare_runnable_artifact(
            request,
            artifact_path=artifact_path,
            classes=classes,
            input_shape=input_shape,
        )
        try:
            deployment = registry.rollback(project_id=project_id)
        except RegistryError:
            _close_candidate(candidate)
            raise
        request.app.state.inference.commit(
            candidate,
            artifact_path=artifact_path,
            classes=classes,
            input_shape=input_shape,
        )
    except RegistryError as exc:
        logger.warning("model rollback rejected project_id=%s error=%s", project_id, exc)
        raise _as_http_error(exc) from exc
    return {
        "deployment": asdict(deployment),
        "inference": _inference_status(request),
    }
