from __future__ import annotations

import hashlib
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

router = APIRouter(prefix="/api/models")

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
    if artifact.kind != "engine":
        request.app.state.inference.disable(
            f"published artifact is not TensorRT engine: {artifact.kind}"
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


@router.get("/projects")
def list_projects(request: Request) -> list[dict[str, Any]]:
    return [asdict(project) for project in _registry(request).list_projects()]


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
    input_shape: str = Form("1x3x640x640"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    registry = _registry(request)
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="model filename is required")
    try:
        kind = _artifact_kind_from_filename(filename)
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
                input_shape=input_shape,
            )
        asset_path = Path(registry.data_dir) / project.name / model_version.version / filename
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(await file.read())
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
        raise _as_http_error(exc) from exc
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
        deployment = registry.publish(
            project_id=project_id,
            artifact_id=payload.artifact_id,
        )
        _load_published_artifact(request, registry, payload.artifact_id)
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return asdict(deployment)


@router.post("/projects/{project_id}/rollback")
def rollback(request: Request, project_id: int) -> dict[str, Any]:
    registry = _registry(request)
    try:
        _require_project(registry, project_id)
        deployment = registry.rollback(project_id=project_id)
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    return asdict(deployment)
