from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from novasight.model_registry import (
    EngineCache,
    ModelRegistry,
    RegistryConflictError,
    RegistryError,
    RegistryNotFoundError,
    RegistryValidationError,
    read_manifest,
)
from novasight.model_registry.fingerprint import sha256_file
from novasight.model_registry.import_model import import_onnx_model


router = APIRouter(prefix="/api/v1/models")


class ModelImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: StrictStr
    model_id: StrictStr | None = None
    version: StrictStr = "imported"
    display_name: StrictStr | None = None
    classes: list[StrictStr] = []
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    precision: StrictStr = "fp16"


class ModelBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: StrictInt | None = None
    artifact_id: StrictInt | None = None
    force: bool = False


@router.post("/import")
def import_model(request: Request, payload: ModelImportRequest) -> dict[str, Any]:
    registry: ModelRegistry = request.app.state.models
    source_path = Path(payload.source_path).expanduser()
    model_id = _safe_model_id(payload.model_id or source_path.stem)
    version_name = _safe_model_id(payload.version or "imported")
    target_dir = Path(registry.data_dir) / model_id / version_name
    try:
        imported = import_onnx_model(
            source_path,
            target_dir=target_dir,
            model_id=model_id,
            display_name=payload.display_name,
            class_names=list(payload.classes) or None,
            confidence_threshold=float(payload.confidence_threshold),
            nms_iou_threshold=float(payload.nms_iou_threshold),
            runtime_precision=str(payload.precision),
        )
        project = _get_or_create_project(registry, model_id, payload.display_name or model_id)
        version = _get_or_create_version(
            registry,
            project_id=project.id,
            version_name=version_name,
            source_path=imported.model_path.name,
            classes=list(imported.manifest.output.class_names),
            input_shape=_shape_label(imported.manifest.input.shape),
        )
        artifact = _get_or_create_artifact(
            registry,
            version_id=version.id,
            path=imported.model_path.name,
            checksum=f"sha256:{sha256_file(imported.model_path)}",
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "project": asdict(project),
        "version": asdict(version),
        "artifact": asdict(artifact),
        "model_path": _relative_registry_path(registry, imported.model_path),
        "manifest_path": _relative_registry_path(registry, imported.manifest_path),
        "model_fingerprint": imported.manifest.model_fingerprint,
        "input": imported.manifest.input.__dict__,
        "output": imported.manifest.output.__dict__,
    }


@router.post("/{project_id}/build")
def build_model_engine(
    request: Request,
    project_id: int,
    payload: ModelBuildRequest | None = None,
) -> dict[str, Any]:
    registry: ModelRegistry = request.app.state.models
    payload = payload or ModelBuildRequest()
    try:
        project = registry.get_project(project_id)
        if project is None:
            raise RegistryNotFoundError(f"unknown project id: {project_id}")
        version = _select_build_version(registry, project_id, payload.version_id)
        source_artifact = _select_build_source_artifact(registry, version.id, payload.artifact_id)
        source_path = Path(registry.data_dir) / project.name / version.version / source_artifact.path
        manifest_path = source_path.with_name("model.manifest.json")
        if not manifest_path.is_file():
            raise RegistryValidationError(f"model manifest missing: {manifest_path}")
        manifest = read_manifest(manifest_path)
        cache_dir = Path(registry.data_dir) / "generated"
        cached_engine = None if payload.force else EngineCache.get_engine_path(manifest, cache_dir=cache_dir)
        cache_hit = cached_engine is not None
        built = False
        if cached_engine is None:
            if not shutil.which("trtexec") and source_path.suffix.lower() == ".onnx":
                job = registry.create_conversion_job(
                    version.id,
                    "engine",
                    _trtexec_command(
                        manifest.input.name,
                        manifest.input.shape,
                        source_path,
                        cache_dir,
                        manifest,
                    ),
                )
                return {
                    "project": asdict(project),
                    "version": asdict(version),
                    "source_artifact": asdict(source_artifact),
                    "status": "pending",
                    "reason": "trtexec not found; engine build has been recorded as a conversion job",
                    "job": asdict(job),
                    "cache_key": EngineCache.compute_cache_key(manifest),
                    "cache_path": str(EngineCache.engine_path_for(manifest, cache_dir=cache_dir)),
                }
            cached_engine = EngineCache.build_engine(
                manifest,
                source_model_path=source_path,
                cache_dir=cache_dir,
            )
            built = True
        artifact_path = source_path.with_name("engine.plan")
        if cached_engine.resolve(strict=False) != artifact_path.resolve(strict=False):
            shutil.copy2(cached_engine, artifact_path)
        engine_artifact = _get_or_create_engine_artifact(
            registry,
            version_id=version.id,
            path=artifact_path.name,
            checksum=f"sha256:{sha256_file(artifact_path)}",
        )
    except RegistryError as exc:
        raise _as_http_error(exc) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "project": asdict(project),
        "version": asdict(version),
        "source_artifact": asdict(source_artifact),
        "artifact": asdict(engine_artifact),
        "status": "ready",
        "built": built,
        "cache_hit": cache_hit,
        "cache_key": EngineCache.compute_cache_key(manifest),
        "cache_path": str(cached_engine),
        "engine_path": _relative_registry_path(registry, artifact_path),
    }


def _get_or_create_project(registry: ModelRegistry, name: str, description: str):
    existing = next((project for project in registry.list_projects() if project.name == name), None)
    if existing is not None:
        return existing
    return registry.create_project(name=name, description=description)


def _get_or_create_version(
    registry: ModelRegistry,
    *,
    project_id: int,
    version_name: str,
    source_path: str,
    classes: list[str],
    input_shape: str,
):
    existing = next(
        (
            version
            for version in registry.list_versions(project_id)
            if version.version == version_name
        ),
        None,
    )
    if existing is not None:
        return existing
    return registry.create_version(
        project_id=project_id,
        version=version_name,
        source_kind="onnx",
        source_path=source_path,
        classes=classes,
        input_shape=input_shape,
    )


def _select_build_version(registry: ModelRegistry, project_id: int, version_id: int | None):
    versions = registry.list_versions(project_id)
    if version_id is not None:
        version = registry.get_version(version_id)
        if version is None:
            raise RegistryNotFoundError(f"unknown version id: {version_id}")
        if version.project_id != project_id:
            raise RegistryValidationError("version does not belong to project")
        return version
    if not versions:
        raise RegistryValidationError("project has no model versions to build")
    return versions[-1]


def _select_build_source_artifact(
    registry: ModelRegistry,
    version_id: int,
    artifact_id: int | None,
):
    artifacts = registry.list_artifacts(version_id)
    if artifact_id is not None:
        artifact = registry.get_artifact(artifact_id)
        if artifact is None:
            raise RegistryNotFoundError(f"unknown artifact id: {artifact_id}")
        if artifact.version_id != version_id:
            raise RegistryValidationError("artifact does not belong to selected version")
        return artifact
    source = next((artifact for artifact in artifacts if artifact.kind == "onnx"), None)
    if source is None:
        raise RegistryValidationError("selected version has no ONNX artifact to build")
    return source


def _trtexec_command(
    input_name: str,
    input_shape: list[int],
    source_path: Path,
    cache_dir: Path,
    manifest: Any,
) -> list[str]:
    engine_path = EngineCache.engine_path_for(manifest, cache_dir=cache_dir)
    return [
        "trtexec",
        f"--onnx={source_path}",
        f"--saveEngine={engine_path}",
        f"--shapes={input_name}:{'x'.join(str(dim) for dim in input_shape)}",
    ]


def _get_or_create_engine_artifact(
    registry: ModelRegistry,
    *,
    version_id: int,
    path: str,
    checksum: str,
):
    existing = next(
        (
            artifact
            for artifact in registry.list_artifacts(version_id)
            if artifact.path == path
        ),
        None,
    )
    if existing is not None:
        if existing.checksum != checksum or existing.status != "ready":
            return registry.update_artifact_status(existing.id, "ready", checksum=checksum)
        return existing
    return registry.create_artifact(
        version_id=version_id,
        kind="engine",
        path=path,
        checksum=checksum,
        status="ready",
    )


def _get_or_create_artifact(
    registry: ModelRegistry,
    *,
    version_id: int,
    path: str,
    checksum: str,
):
    existing = next(
        (
            artifact
            for artifact in registry.list_artifacts(version_id)
            if artifact.path == path
        ),
        None,
    )
    if existing is not None:
        if existing.checksum != checksum or existing.status != "ready":
            return registry.update_artifact_status(existing.id, "ready", checksum=checksum)
        return existing
    return registry.create_artifact(
        version_id=version_id,
        kind="onnx",
        path=path,
        checksum=checksum,
        status="ready",
    )


def _safe_model_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RegistryValidationError("model id must not be empty")
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise RegistryValidationError("model id must be a safe path component")
    return text


def _shape_label(shape: list[int]) -> str:
    return "x".join(str(int(dim)) for dim in shape)


def _relative_registry_path(registry: ModelRegistry, path: Path) -> str:
    try:
        return Path(path).resolve(strict=False).relative_to(
            Path(registry.data_dir).resolve(strict=False)
        ).as_posix()
    except ValueError:
        return str(path)


def _as_http_error(exc: RegistryError) -> HTTPException:
    if isinstance(exc, RegistryNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (RegistryConflictError, RegistryValidationError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


__all__ = ["ModelBuildRequest", "ModelImportRequest", "router"]
