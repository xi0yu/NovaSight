from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictStr

from novasight.model_registry import (
    ModelRegistry,
    RegistryConflictError,
    RegistryError,
    RegistryNotFoundError,
    RegistryValidationError,
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


__all__ = ["ModelImportRequest", "router"]
