from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
import threading
from typing import Any, Iterator, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from novasight.model_ingress import (
    EngineInspector,
    InferenceConfigBuilder,
    ModelProbe,
    ModelProfile,
    ModelProfileResolver,
    ModelProfileStore,
    ModelStatus,
    PostprocessProfile,
    model_profile_validation_fingerprint,
)
from novasight.model_registry import ModelRegistry, write_manifest


router = APIRouter(prefix="/api/models")
_MODEL_INGRESS_LOCK = threading.RLock()


def serialized_model_operation(handler):
    @wraps(handler)
    def wrapped(*args, **kwargs):
        with _MODEL_INGRESS_LOCK:
            return handler(*args, **kwargs)

    return wrapped


class ModelProfileConfigureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color_format: StrictStr
    scale: float = Field(gt=0.0)
    offsets: list[float] = Field(default_factory=list)
    mean: list[float] = Field(default_factory=list)
    std: list[float] = Field(default_factory=list)
    resize_mode: StrictStr
    symmetric_padding: StrictBool = False
    padding_value: float = 0.0
    parser_type: StrictStr
    class_count: StrictInt = Field(gt=0)
    labels: list[StrictStr]
    bbox_format: StrictStr
    has_objectness: StrictBool
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: StrictInt = Field(default=300, gt=0)


class ModelProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_mode: Literal["fixed", "latest"] = "fixed"


@router.post("/artifacts/{artifact_id}/inspect")
@serialized_model_operation
def inspect_engine_artifact(request: Request, artifact_id: int) -> dict[str, Any]:
    registry, artifact, version, project, engine_path = _artifact_context(
        request,
        artifact_id,
    )
    inspection = EngineInspector().inspect(engine_path)
    profile = ModelProfileResolver().resolve(
        engine_path=engine_path,
        inspection=inspection,
        display_name=project.name,
    )
    profile_path = ModelProfileStore().write(profile)
    registry.update_artifact_status(
        artifact.id,
        "pending" if profile.status is not ModelStatus.INCOMPATIBLE else "failed",
        checksum=profile.engine.sha256,
    )
    if profile.input.runtime_shape:
        registry.update_version_input_shape(
            version.id,
            "x".join(str(value) for value in profile.input.runtime_shape),
        )
    return {
        "artifact_id": artifact.id,
        "profile_path": _profile_path_payload(registry, profile_path),
        "profile": _payload(profile),
    }


@router.get("/artifacts/{artifact_id}/profile")
def get_engine_profile(request: Request, artifact_id: int) -> dict[str, Any]:
    registry, artifact, _version, _project, engine_path = _artifact_context(
        request,
        artifact_id,
    )
    profile_path = ModelProfileStore().path_for_engine(engine_path)
    if not profile_path.is_file():
        raise HTTPException(status_code=404, detail="model profile has not been inspected")
    profile = ModelProfileStore().load(profile_path)
    return {
        "artifact_id": artifact.id,
        "profile_path": _profile_path_payload(registry, profile_path),
        "profile": _payload(profile),
    }


@router.put("/artifacts/{artifact_id}/profile")
@serialized_model_operation
def configure_engine_artifact(
    request: Request,
    artifact_id: int,
    payload: ModelProfileConfigureRequest,
) -> dict[str, Any]:
    registry, artifact, version, _project, engine_path = _artifact_context(
        request,
        artifact_id,
    )
    store = ModelProfileStore()
    profile_path = store.path_for_engine(engine_path)
    if not profile_path.is_file():
        raise HTTPException(status_code=409, detail="inspect the TensorRT engine first")
    profile = store.load(profile_path)
    if profile.status is ModelStatus.UNINSPECTED:
        raise HTTPException(
            status_code=409,
            detail="engine content changed; inspect it again before configuration",
        )
    if profile.status is ModelStatus.ACTIVE:
        raise HTTPException(
            status_code=409,
            detail="stop or switch away from the active model before changing model semantics",
        )
    try:
        configured = ModelProfileResolver().configure(
            profile,
            color_format=payload.color_format,
            scale=float(payload.scale),
            offsets=payload.offsets,
            mean=payload.mean,
            std=payload.std,
            resize_mode=payload.resize_mode,
            symmetric_padding=payload.symmetric_padding,
            padding_value=float(payload.padding_value),
            parser_type=payload.parser_type,
            class_count=int(payload.class_count),
            labels=payload.labels,
            bbox_format=payload.bbox_format,
            has_objectness=payload.has_objectness,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    configured = replace(
        configured,
        postprocess=PostprocessProfile(
            confidence_threshold=float(payload.confidence_threshold),
            nms_threshold=float(payload.nms_threshold),
            max_detections=int(payload.max_detections),
        ),
    )
    store.write(configured, profile_path)
    registry.update_version_classes(version.id, list(configured.labels))
    registry.update_version_input_shape(
        version.id,
        "x".join(str(value) for value in configured.input.runtime_shape),
    )
    registry.update_artifact_status(
        artifact.id,
        "pending",
        checksum=configured.engine.sha256,
    )
    return {
        "artifact_id": artifact.id,
        "profile_path": _profile_path_payload(registry, profile_path),
        "profile": _payload(configured),
    }


@router.post("/artifacts/{artifact_id}/probe")
@serialized_model_operation
def probe_engine_artifact(
    request: Request,
    artifact_id: int,
    payload: ModelProbeRequest | None = None,
) -> dict[str, Any]:
    registry, artifact, _version, _project, engine_path = _artifact_context(
        request,
        artifact_id,
    )
    store = ModelProfileStore()
    profile_path = store.path_for_engine(engine_path)
    if not profile_path.is_file():
        raise HTTPException(status_code=409, detail="inspect and configure the model first")
    profile = store.load(profile_path)
    if profile.status is not ModelStatus.READY_FOR_PROBE:
        raise HTTPException(
            status_code=409,
            detail=f"model profile is not ready for diagnostics: {profile.status.value}",
        )
    input_mode = payload.input_mode if payload is not None else "fixed"
    validated, report = ModelProbe().run(
        profile,
        inference=request.app.state.inference,
        isolation=_diagnostic_isolation(request),
        frame_supplier=(
            (lambda: _latest_diagnostic_frame(request))
            if input_mode == "latest"
            else None
        ),
    )
    if validated.status is ModelStatus.VALIDATED:
        try:
            runtime_config = InferenceConfigBuilder().build(
                validated,
                parser_library_path=Path(
                    request.app.state.config.inference.deepstream_parser_library
                ),
            )
        except ValueError as exc:
            validated = replace(
                validated,
                status=ModelStatus.INVALID,
                validation=replace(
                    validated.validation,
                    status="invalid",
                    issues=(*validated.validation.issues, "RUNTIME_CONFIG_INVALID"),
                ),
            )
            report = replace(
                report,
                status="invalid",
                issues=(
                    *report.issues,
                    _runtime_config_issue(str(exc)),
                ),
            )
        else:
            write_manifest(
                runtime_config.manifest,
                engine_path.with_name("model.manifest.json"),
            )
    store.write(validated, profile_path)
    registry.update_artifact_status(
        artifact.id,
        "ready" if validated.status is ModelStatus.VALIDATED else "failed",
        checksum=validated.engine.sha256,
    )
    return {
        "artifact_id": artifact.id,
        "profile_path": _profile_path_payload(registry, profile_path),
        "profile": _payload(validated),
        "report": _payload(report),
    }


def load_validated_profile(engine_path: Path) -> ModelProfile:
    store = ModelProfileStore()
    profile_path = store.path_for_engine(engine_path)
    if not profile_path.is_file():
        raise ValueError("TensorRT engine has no ModelProfile; inspect and diagnose it first")
    profile = store.load(profile_path)
    if profile.status not in {ModelStatus.VALIDATED, ModelStatus.ACTIVE}:
        raise ValueError(
            "TensorRT engine is not validated for control activation "
            f"(status={profile.status.value})"
        )
    validation = profile.validation
    if not (
        validation.engine_execution_ok
        and validation.decoder_ok
        and validation.nms_ok
        and validation.detection_batch_ok
    ):
        raise ValueError("TensorRT ModelProfile validation report is incomplete")
    expected_fingerprint = model_profile_validation_fingerprint(profile)
    if validation.profile_fingerprint != expected_fingerprint:
        raise ValueError(
            "TensorRT ModelProfile semantics changed after validation; run diagnostics again"
        )
    return profile


def set_profile_activation(engine_path: Path, *, active: bool) -> ModelProfile:
    store = ModelProfileStore()
    profile = load_validated_profile(engine_path)
    updated = replace(
        profile,
        status=ModelStatus.ACTIVE if active else ModelStatus.VALIDATED,
    )
    store.write(updated, store.path_for_engine(engine_path))
    return updated


@contextmanager
def _diagnostic_isolation(request: Request) -> Iterator[None]:
    runtime = getattr(request.app.state, "runtime", None)
    pipeline = getattr(runtime, "pipeline", None) if runtime is not None else None
    was_running = bool(pipeline is not None and getattr(pipeline, "running", False))
    if was_running:
        pipeline.stop()
    elif runtime is not None:
        cancel = getattr(runtime, "cancel_control", None)
        if callable(cancel):
            cancel("MODEL_DIAGNOSTIC")
    try:
        yield
    finally:
        if was_running:
            pipeline.start()


def _latest_diagnostic_frame(request: Request) -> Any:
    capture = getattr(request.app.state, "capture", None)
    latest_frame = getattr(capture, "get_latest_preview_frame", None)
    if callable(latest_frame):
        frame = latest_frame()
    else:
        session = getattr(capture, "session", None)
        latest_frame = getattr(session, "latest_frame", None)
        frame = latest_frame(timeout_s=0.0) if callable(latest_frame) else None
    if frame is None:
        raise RuntimeError("latest ROI diagnostic requested but capture has no current frame")
    return frame


def _artifact_context(
    request: Request,
    artifact_id: int,
) -> tuple[ModelRegistry, Any, Any, Any, Path]:
    registry: ModelRegistry = request.app.state.models
    artifact = registry.get_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"unknown artifact id: {artifact_id}")
    if artifact.kind != "engine":
        raise HTTPException(status_code=400, detail="model ingress requires a TensorRT .engine")
    version = registry.get_version(artifact.version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="model version is missing")
    project = registry.get_project(version.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="model project is missing")
    engine_path = Path(registry.data_dir) / project.name / version.version / artifact.path
    return registry, artifact, version, project, engine_path


def _runtime_config_issue(message: str):
    from novasight.model_ingress import ValidationIssue

    return ValidationIssue(
        code="RUNTIME_CONFIG_INVALID",
        stage="config",
        message=message,
    )


def _profile_path_payload(registry: ModelRegistry, profile_path: Path) -> str:
    try:
        return profile_path.relative_to(Path(registry.data_dir)).as_posix()
    except ValueError:
        return profile_path.name


def _payload(value: Any) -> Any:
    return jsonable_encoder(asdict(value))


__all__ = [
    "ModelProfileConfigureRequest",
    "ModelProbeRequest",
    "configure_engine_artifact",
    "get_engine_profile",
    "inspect_engine_artifact",
    "load_validated_profile",
    "probe_engine_artifact",
    "router",
    "serialized_model_operation",
    "set_profile_activation",
]
