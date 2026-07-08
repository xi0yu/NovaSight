from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from novasight.capture.service import CaptureService
from novasight.config import RuntimeConfig, load_runtime_config
from novasight.executors import ExecutorRegistry
from novasight.hardware import create_hardware_box
from novasight.inference import InferenceRuntime
from novasight.inference.jetson import create_gpu_resource_preprocessor
from novasight.instance_lock import InstanceLock
from novasight.license import LicenseStore
from novasight.model_registry import ModelRegistry
from novasight.runtime import ControlFrameCsvRecorder, ControlFrameParquetRecorder, RuntimeService
from novasight.systemd import SystemdNotifier, watchdog_interval_from_env

from .routes_capture import router as capture_router
from .routes_device import router as device_router
from .routes_executors import router as executors_router
from .routes_health import router as health_router
from .routes_model_import import router as model_import_router
from .routes_models import router as models_router
from .routes_runtime import router as runtime_router
from .routes_system import router as system_router
from .routes_websocket import router as websocket_router

logger = logging.getLogger("novasight.api.app")


OPEN_API_PATHS = {
    "/healthz",
    "/api/license",
    "/api/license/activate",
    "/api/config/schema",
}


def _is_open_path(path: str) -> bool:
    if path in OPEN_API_PATHS:
        return True
    return not path.startswith("/api/")


def create_app(
    data_dir: Path | str = "data",
    config_path: Path | str = "config/novasight.yaml",
    config: RuntimeConfig | None = None,
) -> FastAPI:
    app = FastAPI(title="NovaSight")
    data_path = Path(data_dir)
    config = config or load_runtime_config(config_path)
    models = ModelRegistry(
        db_path=data_path / "novasight.db",
        data_dir=data_path / "models",
    )
    executors = ExecutorRegistry.from_config(config)
    hardware = create_hardware_box(config)
    capture = CaptureService(
        config.capture,
        roi_size=config.roi.size,
        roi_offset_x=config.roi.offset_x,
        roi_offset_y=config.roi.offset_y,
    )
    inference = InferenceRuntime(
        gpu_preprocessor=create_gpu_resource_preprocessor(config),
    )
    inference.configure(
        confidence_threshold=config.inference.confidence_threshold,
        nms_threshold=config.inference.nms_threshold,
    )
    _load_active_model(models, inference)
    runtime = RuntimeService(
        config=config,
        models=models,
        executors=executors,
        hardware=hardware,
        capture=capture,
        inference=inference,
        recorder=_create_control_frame_recorder(config, data_path),
    )
    systemd_notifier = SystemdNotifier(interval_s=watchdog_interval_from_env())
    instance_lock = InstanceLock()

    app.state.config = config
    app.state.config_path = Path(config_path)
    app.state.models = models
    app.state.executors = executors
    app.state.hardware = hardware
    app.state.license = LicenseStore(data_path / "license.json")
    app.state.capture = capture
    app.state.inference = inference
    app.state.runtime = runtime
    app.state.systemd_notifier = systemd_notifier
    app.state.instance_lock = instance_lock

    @app.on_event("startup")
    def start_process_lifecycle() -> None:
        instance_lock.acquire()
        systemd_notifier.start()

    @app.on_event("shutdown")
    def stop_process_lifecycle() -> None:
        systemd_notifier.stop()
        instance_lock.release()

    @app.middleware("http")
    async def require_license(request: Request, call_next):
        if request.method == "OPTIONS" or _is_open_path(request.url.path):
            return await call_next(request)
        status = request.app.state.license.status()
        if not status.configured or not status.valid:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "license required",
                    "license": app.state.license.asdict(),
                },
            )
        return await call_next(request)

    app.include_router(health_router)
    app.include_router(device_router)
    app.include_router(capture_router)
    app.include_router(runtime_router)
    app.include_router(models_router)
    app.include_router(model_import_router)
    app.include_router(executors_router)
    app.include_router(system_router)
    app.include_router(websocket_router)
    return app


def _create_control_frame_recorder(config: RuntimeConfig, data_path: Path):
    consumers = config.consumers
    recording_format = str(consumers.recording_format or "csv").lower()
    path = _recording_path(data_path, format=recording_format, configured=consumers.recording_path)
    if recording_format == "csv":
        return ControlFrameCsvRecorder(path)
    if recording_format == "parquet":
        return ControlFrameParquetRecorder(path)
    raise ValueError("runtime config key 'consumers.recording_format' must be csv or parquet")


def _recording_path(data_path: Path, *, format: str, configured: str) -> Path:
    if configured.strip():
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate
        return data_path / candidate
    suffix = "parquet" if format == "parquet" else "csv"
    return data_path / "recordings" / f"control_frames.{suffix}"


def _load_active_model(models: ModelRegistry, inference: InferenceRuntime) -> None:
    deployment = models.get_active_deployment()
    if deployment is None:
        return
    artifact = models.get_artifact(deployment.artifact_id)
    if artifact is None:
        inference.disable(f"active deployment artifact not found: {deployment.artifact_id}")
        return
    if artifact.kind not in {"onnx", "engine"}:
        inference.disable(f"active artifact is not runnable inference artifact: {artifact.kind}")
        return
    version = models.get_version(artifact.version_id)
    if version is None:
        inference.disable(f"active artifact version not found: {artifact.version_id}")
        return
    project = models.get_project(version.project_id)
    if project is None:
        inference.disable(f"active artifact project not found: {version.project_id}")
        return
    artifact_path = Path(models.data_dir) / project.name / version.version / artifact.path
    inference.load(artifact_path, list(version.classes), version.input_shape)
    status = inference.status()
    if status.get("loaded"):
        logger.info("active model loaded artifact=%s", artifact_path)
    else:
        logger.warning("active model failed to load artifact=%s reason=%s", artifact_path, status.get("reason"))
