from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from novasight.model_registry import ModelRegistry

router = APIRouter(prefix="/api/models")


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


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.models


def _as_http_error(exc: ValueError) -> HTTPException:
    message = str(exc)
    if message.startswith("unknown "):
        return HTTPException(status_code=404, detail=message)
    return HTTPException(status_code=400, detail=message)


def _require_project(registry: ModelRegistry, project_id: int) -> None:
    if registry.get_project(project_id) is None:
        raise ValueError(f"unknown project id: {project_id}")


def _require_version(registry: ModelRegistry, version_id: int) -> None:
    if registry.get_version(version_id) is None:
        raise ValueError(f"unknown version id: {version_id}")


@router.get("/projects")
def list_projects(request: Request) -> list[dict[str, Any]]:
    return [asdict(project) for project in _registry(request).list_projects()]


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
    except ValueError as exc:
        raise _as_http_error(exc) from exc
    return asdict(project)


@router.get("/projects/{project_id}/versions")
def list_versions(request: Request, project_id: int) -> list[dict[str, Any]]:
    registry = _registry(request)
    try:
        _require_project(registry, project_id)
    except ValueError as exc:
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
    except ValueError as exc:
        raise _as_http_error(exc) from exc
    return asdict(version)


@router.get("/versions/{version_id}/artifacts")
def list_artifacts(request: Request, version_id: int) -> list[dict[str, Any]]:
    registry = _registry(request)
    try:
        _require_version(registry, version_id)
    except ValueError as exc:
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
    except ValueError as exc:
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
        except ValueError as exc:
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
    except ValueError as exc:
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
    except ValueError as exc:
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
    except ValueError as exc:
        raise _as_http_error(exc) from exc
    return asdict(deployment)


@router.post("/projects/{project_id}/rollback")
def rollback(request: Request, project_id: int) -> dict[str, Any]:
    registry = _registry(request)
    try:
        _require_project(registry, project_id)
        deployment = registry.rollback(project_id=project_id)
    except ValueError as exc:
        raise _as_http_error(exc) from exc
    return asdict(deployment)
