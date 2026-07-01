from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from novasight.capture.service import CaptureService
from novasight.config import RuntimeConfig, load_runtime_config
from novasight.executors import ExecutorRegistry
from novasight.inference import InferenceRuntime
from novasight.model_registry import ModelRegistry
from novasight.plugins import PluginRuntime
from novasight.runtime import RuntimeService

from .routes_capture import router as capture_router
from .routes_executors import router as executors_router
from .routes_health import router as health_router
from .routes_models import router as models_router
from .routes_plugins import router as plugins_router


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
    plugins = PluginRuntime.with_builtin_plugins()
    executors = ExecutorRegistry.with_builtin_executors(
        default=config.executor.default
    )
    capture = CaptureService(config.capture)
    inference = InferenceRuntime()
    runtime = RuntimeService(
        config=config,
        models=models,
        plugins=plugins,
        executors=executors,
        capture=capture,
        inference=inference,
    )

    app.state.config = config
    app.state.models = models
    app.state.plugins = plugins
    app.state.executors = executors
    app.state.capture = capture
    app.state.inference = inference
    app.state.runtime = runtime

    app.include_router(health_router)
    app.include_router(capture_router)
    app.include_router(models_router)
    app.include_router(plugins_router)
    app.include_router(executors_router)
    return app
