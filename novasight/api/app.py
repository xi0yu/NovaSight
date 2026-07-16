from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from novasight.capture.service import CaptureService
from novasight.config import RuntimeConfig, load_runtime_config
from novasight.crosshair import CrosshairSystem
from novasight.executors import ExecutorRegistry
from novasight.inference import InferenceRuntime
from novasight.inference.jetson import create_gpu_resource_preprocessor
from novasight.instance_lock import InstanceLock
from novasight.license import LicenseStore
from novasight.model_registry import ModelRegistry
from novasight.runtime import (
    CallbackRuntimeLifecycle,
    ControlFrameCsvRecorder,
    ControlFrameParquetRecorder,
    RuntimePowerSupervisor,
    RuntimeService,
    StatusHub,
)
from novasight.systemd import SystemdNotifier, watchdog_interval_from_env

from .routes_capture import router as capture_router
from .routes_capture_v1 import router as capture_v1_router
from .routes_control import router as control_router
from .routes_crosshair import router as crosshair_router
from .routes_device import router as device_router
from .routes_executors import router as executors_router
from .routes_health import router as health_router
from .routes_model_ingress import load_validated_profile, router as model_ingress_router
from .routes_models import router as models_router
from .routes_runtime import (
    _start_runtime_pipeline_core,
    _stop_runtime_pipeline_core,
    router as runtime_router,
)
from .routes_status import router as status_router
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
    _install_studio_cors(app, config)
    models = ModelRegistry(
        db_path=data_path / "novasight.db",
        data_dir=data_path / "models",
    )
    executors = ExecutorRegistry.from_config(config)
    capture = CaptureService(
        config.capture,
        roi_size=config.roi.size,
    )
    inference = InferenceRuntime(
        gpu_preprocessor=create_gpu_resource_preprocessor(config),
    )
    inference.configure(
        confidence_threshold=config.inference.confidence_threshold,
        nms_threshold=config.inference.nms_threshold,
    )
    _load_active_model(models, inference, config)
    runtime = RuntimeService(
        config=config,
        models=models,
        executors=executors,
        capture=capture,
        inference=inference,
        recorder=_create_control_frame_recorder(config, data_path),
        crosshair=CrosshairSystem(
            config.crosshair,
            template_path=data_path / "crosshair" / "template.json",
        ),
    )
    systemd_notifier = SystemdNotifier(interval_s=watchdog_interval_from_env())
    instance_lock = InstanceLock()

    app.state.config = config
    app.state.config_path = Path(config_path)
    app.state.models = models
    app.state.executors = executors
    app.state.license = LicenseStore(data_path / "license.json")
    app.state.capture = capture
    app.state.inference = inference
    app.state.runtime = runtime
    app.state.status_hub = StatusHub(runtime)
    app.state.systemd_notifier = systemd_notifier
    app.state.instance_lock = instance_lock
    app.state.runtime_reconfiguration_lock = threading.RLock()
    app.state.kmnet_auto_connect_thread = None
    app.state.capture_auto_restore_thread = None
    runtime_power = RuntimePowerSupervisor(
        lifecycle=CallbackRuntimeLifecycle(
            is_running=lambda: bool(
                runtime.pipeline is not None and runtime.pipeline.running
            ),
            start=lambda _reason: _start_runtime_pipeline_core(app),
            stop=lambda reason: _stop_runtime_pipeline_core(runtime, reason),
        ),
        enabled=config.power_saving.host_presence_enabled,
        target_host_id=config.power_saving.target_host_id,
        heartbeat_timeout_s=config.power_saving.heartbeat_timeout_s,
        offline_grace_s=config.power_saving.offline_grace_s,
        auto_resume=config.power_saving.auto_resume,
    )
    app.state.runtime_power = runtime_power
    runtime.power_supervisor = runtime_power

    @app.on_event("startup")
    def start_process_lifecycle() -> None:
        logger.info("application startup: acquiring instance lock")
        instance_lock.acquire()
        logger.info("application startup: instance lock acquired")
        systemd_notifier.start()
        logger.info("application startup: systemd notifier started enabled=%s", systemd_notifier.enabled)
        app.state.kmnet_auto_connect_thread = _start_auto_connect_kmnet(executors, config)
        app.state.capture_auto_restore_thread = _start_auto_restore_capture(
            capture,
            config,
        )
        runtime_power.start_monitoring()
        logger.info("application startup: lifecycle ready")

    @app.on_event("shutdown")
    def stop_process_lifecycle() -> None:
        runtime_power.close()
        app.state.status_hub.close()
        _disconnect_kmnet(app.state.executors)
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
    app.include_router(capture_v1_router)
    app.include_router(control_router)
    app.include_router(crosshair_router)
    app.include_router(runtime_router)
    app.include_router(status_router)
    app.include_router(models_router)
    app.include_router(model_ingress_router)
    app.include_router(executors_router)
    app.include_router(system_router)
    app.include_router(websocket_router)
    return app


def _auto_connect_kmnet(executors: ExecutorRegistry, config: RuntimeConfig) -> None:
    if not bool(getattr(config.hardware, "auto_connect", True)):
        logger.info("kmNet auto-connect disabled")
        return
    kmnet = executors.executors.get("kmnet")
    connect = getattr(kmnet, "connect", None)
    if not callable(connect):
        logger.warning("kmNet auto-connect unavailable: executor has no connect method")
        return
    try:
        status = connect()
    except Exception as exc:
        logger.warning("kmNet auto-connect failed: %s", exc)
        return
    if status.get("connected") is True:
        logger.info(
            "kmNet auto-connected host=%s port=%s monitor_port=%s",
            status.get("host"),
            status.get("port"),
            status.get("monitor_port"),
        )
        return
    logger.warning(
        "kmNet auto-connect did not connect: %s",
        status.get("last_error") or status,
    )


def _start_auto_connect_kmnet(
    executors: ExecutorRegistry,
    config: RuntimeConfig,
) -> threading.Thread | None:
    if not bool(getattr(config.hardware, "auto_connect", True)):
        logger.info("kmNet auto-connect disabled")
        return None
    thread = threading.Thread(
        target=_auto_connect_kmnet,
        args=(executors, config),
        name="novasight-kmnet-auto-connect",
        daemon=True,
    )
    thread.start()
    logger.info("kmNet auto-connect scheduled in background")
    return thread


def _disconnect_kmnet(executors: ExecutorRegistry) -> None:
    kmnet = executors.executors.get("kmnet")
    disconnect = getattr(kmnet, "disconnect", None)
    if not callable(disconnect):
        return
    try:
        disconnect()
    except Exception as exc:
        logger.warning("kmNet shutdown disconnect failed: %s", exc)


def _start_auto_restore_capture(
    capture: CaptureService,
    config: RuntimeConfig,
) -> threading.Thread | None:
    source_default = str(getattr(getattr(config, "source", None), "default", "") or "").strip()
    capture_cfg = getattr(config, "capture", None)
    device = str(getattr(capture_cfg, "device", "") or "").strip()
    if source_default != "capture" or not device:
        return None
    thread = threading.Thread(
        target=_auto_restore_capture,
        args=(capture, config),
        name="novasight-capture-profile-restore",
        daemon=True,
    )
    thread.start()
    logger.info("capture profile restore scheduled device=%s", device)
    return thread


def _auto_restore_capture(
    capture: CaptureService,
    config: RuntimeConfig,
) -> None:
    source_default = str(getattr(getattr(config, "source", None), "default", "") or "").strip()
    if source_default != "capture":
        return
    capture_cfg = getattr(config, "capture", None)
    device = str(getattr(capture_cfg, "device", "") or "").strip()
    if not device:
        return
    logger.info("capture profile restore beginning device=%s", device)
    try:
        state = capture.configure_profile_only(
            device,
            preference=str(getattr(capture_cfg, "preference", "manual") or "manual"),
            pixel_format=getattr(capture_cfg, "pixel_format", None),
            width=getattr(capture_cfg, "width", None),
            height=getattr(capture_cfg, "height", None),
            fps=getattr(capture_cfg, "fps", None),
        )
    except Exception as exc:
        logger.warning("DeepStream capture profile selection failed device=%s error=%s", device, exc)
        return
    if getattr(state, "available", False) is True:
        logger.info(
            "DeepStream capture profile selected device=%s profile=%s",
            device,
            f"{state.profile.pixel_format} {state.profile.width}x{state.profile.height}@{state.profile.fps}"
            if getattr(state, "profile", None) is not None
            else "<unknown>",
        )
        logger.info("capture profile restored; DeepStream mainline remains idle")
    else:
        logger.info(
            "DeepStream capture profile unavailable device=%s reason=%s",
            device,
            getattr(state, "last_error", "unknown"),
        )

def _install_studio_cors(app: FastAPI, config: RuntimeConfig) -> None:
    studio_port = int(getattr(getattr(config, "web", None), "port", 5174))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://tauri.localhost",
            "https://tauri.localhost",
            "tauri://localhost",
            f"http://localhost:{studio_port}",
            f"http://127.0.0.1:{studio_port}",
        ],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _create_control_frame_recorder(config: RuntimeConfig, data_path: Path):
    consumers = config.consumers
    if not bool(consumers.recording):
        return None
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


def _load_active_model(
    models: ModelRegistry,
    inference: InferenceRuntime,
    config: RuntimeConfig,
) -> None:
    deployment = models.get_active_deployment()
    if deployment is None:
        if str(config.inference.backend).lower() == "deepstream_nvinfer":
            inference.unload("TensorRT engine ownership delegated to DeepStream nvinfer")
        return
    artifact = models.get_artifact(deployment.artifact_id)
    if artifact is None:
        inference.disable(f"active deployment artifact not found: {deployment.artifact_id}")
        return
    if artifact.kind != "engine":
        inference.disable(
            "active artifact must be a TensorRT .engine; "
            f"CPU/ONNX fallback is disabled for runtime inference: {artifact.kind}"
        )
        return
    version = models.get_version(artifact.version_id)
    if version is None:
        inference.disable(f"active artifact version not found: {artifact.version_id}")
        return
    project = models.get_project(version.project_id)
    if project is None:
        inference.disable(f"active artifact project not found: {version.project_id}")
        return
    artifact_path = models.resolve_artifact_path(artifact)
    if str(config.inference.backend).lower() == "deepstream_nvinfer":
        inference.unload("TensorRT engine ownership delegated to DeepStream nvinfer")
        logger.info(
            "active DeepStream engine selected; runtime contract will be verified on pipeline start "
            "artifact=%s",
            artifact_path,
        )
        return
    try:
        profile = load_validated_profile(artifact_path)
        candidate, candidate_status = inference.prepare_profile(
            profile,
            diagnostic=False,
        )
        if (
            candidate_status.get("loaded") is not True
            or candidate_status.get("warmed") is not True
        ):
            close = getattr(candidate, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                str(candidate_status.get("reason") or "validated model failed to load")
            )
        inference.commit(
            candidate,
            artifact_path=artifact_path,
            classes=list(profile.labels),
            input_shape="x".join(str(value) for value in profile.input.runtime_shape),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        inference.disable(f"active TensorRT ModelProfile rejected: {exc}")
        return
    status = inference.status()
    if status.get("loaded"):
        logger.info("active model loaded artifact=%s", artifact_path)
    else:
        logger.warning("active model failed to load artifact=%s reason=%s", artifact_path, status.get("reason"))
