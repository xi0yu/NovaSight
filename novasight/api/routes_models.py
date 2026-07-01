from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from novasight.model_registry import ModelRegistry

router = APIRouter(prefix="/api/models")


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.models


def _as_http_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _version_asset_dir(registry: ModelRegistry, version_id: int) -> Path:
    for project in registry.list_projects():
        for version in registry.list_versions(project.id):
            if version.id == version_id:
                return registry.data_dir / project.name / version.version
    raise ValueError(f"unknown version id: {version_id}")


def _route_artifact_path(registry: ModelRegistry, version_id: int, path: str) -> str:
    raw_path = Path(path)
    if not raw_path.is_absolute():
        return path
    return str(_version_asset_dir(registry, version_id) / raw_path.name)


@router.get("/projects")
def list_projects(request: Request) -> list[dict[str, Any]]:
    return [asdict(project) for project in _registry(request).list_projects()]


@router.post("/projects")
def create_project(
    request: Request,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        project = _registry(request).create_project(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
        )
    except (KeyError, ValueError) as exc:
        raise _as_http_error(ValueError(str(exc))) from exc
    return asdict(project)


@router.get("/projects/{project_id}/versions")
def list_versions(request: Request, project_id: int) -> list[dict[str, Any]]:
    return [asdict(version) for version in _registry(request).list_versions(project_id)]


@router.post("/projects/{project_id}/versions")
def create_version(
    request: Request,
    project_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        version = _registry(request).create_version(
            project_id=project_id,
            version=str(payload["version"]),
            source_kind=str(payload["source_kind"]),
            source_path=str(payload["source_path"]),
            classes=list(payload["classes"]),
            input_shape=str(payload["input_shape"]),
        )
    except (KeyError, ValueError) as exc:
        raise _as_http_error(ValueError(str(exc))) from exc
    return asdict(version)


@router.get("/versions/{version_id}/artifacts")
def list_artifacts(request: Request, version_id: int) -> list[dict[str, Any]]:
    return [asdict(artifact) for artifact in _registry(request).list_artifacts(version_id)]


@router.post("/versions/{version_id}/artifacts")
def create_artifact(
    request: Request,
    version_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    registry = _registry(request)
    try:
        path = _route_artifact_path(registry, version_id, str(payload["path"]))
        artifact = registry.create_artifact(
            version_id=version_id,
            kind=str(payload["kind"]),
            path=path,
            checksum=str(payload["checksum"]),
            status=str(payload["status"]),
        )
    except (KeyError, ValueError) as exc:
        raise _as_http_error(ValueError(str(exc))) from exc
    return asdict(artifact)


@router.get("/jobs")
def list_conversion_jobs(
    request: Request,
    version_id: int | None = None,
) -> list[dict[str, Any]]:
    return [
        asdict(job)
        for job in _registry(request).list_conversion_jobs(version_id=version_id)
    ]


@router.post("/versions/{version_id}/jobs")
def create_conversion_job(
    request: Request,
    version_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        job = _registry(request).create_conversion_job(
            version_id=version_id,
            target_kind=str(payload["target_kind"]),
            command=list(payload["command"]),
        )
    except (KeyError, ValueError) as exc:
        raise _as_http_error(ValueError(str(exc))) from exc
    return asdict(job)


@router.get("/jobs/{job_id}")
def get_conversion_job(request: Request, job_id: int) -> dict[str, Any]:
    job = _registry(request).get_conversion_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown conversion job id: {job_id}")
    return asdict(job)


@router.post("/jobs/{job_id}/finish")
def finish_conversion_job(
    request: Request,
    job_id: int,
    payload: dict[str, Any],
) -> dict[str, bool]:
    try:
        _registry(request).finish_conversion_job(
            job_id=job_id,
            status=str(payload["status"]),
            log=str(payload.get("log", "")),
        )
    except (KeyError, ValueError) as exc:
        raise _as_http_error(ValueError(str(exc))) from exc
    return {"ok": True}


@router.post("/projects/{project_id}/publish")
def publish(
    request: Request,
    project_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        deployment = _registry(request).publish(
            project_id=project_id,
            artifact_id=int(payload["artifact_id"]),
        )
    except (KeyError, ValueError) as exc:
        raise _as_http_error(ValueError(str(exc))) from exc
    return asdict(deployment)


@router.post("/projects/{project_id}/rollback")
def rollback(request: Request, project_id: int) -> dict[str, Any]:
    try:
        deployment = _registry(request).rollback(project_id=project_id)
    except ValueError as exc:
        raise _as_http_error(exc) from exc
    return asdict(deployment)
