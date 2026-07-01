from __future__ import annotations

from dataclasses import asdict
from typing import Any

from novasight.config import RuntimeConfig
from novasight.executors import ExecutorRegistry
from novasight.model_registry import ModelRegistry
from novasight.plugins import FrameContext, PluginRuntime

from .state import RuntimeFrameResult, RuntimeState


class RuntimeService:
    def __init__(
        self,
        config: RuntimeConfig,
        models: ModelRegistry,
        plugins: PluginRuntime,
        executors: ExecutorRegistry,
        capture: Any | None = None,
        inference: Any | None = None,
    ) -> None:
        self.config = config
        self.models = models
        self.plugins = plugins
        self.executors = executors
        self.capture = capture
        self.inference = inference
        self.running = False

    def state(self) -> RuntimeState:
        capture_state = getattr(self, "capture", None)
        inference_state = getattr(self, "inference", None)
        return RuntimeState(
            running=self.running,
            source=self.config.source.default,
            active_model=self._active_model(),
            executor=self.executors.status(),
            capture=asdict(capture_state.state) if capture_state is not None else {},
            inference=(
                inference_state.status()
                if inference_state is not None
                else {"available": False}
            ),
        )

    def process_frame(self, context: FrameContext) -> RuntimeFrameResult:
        plugin_batch = self.plugins.process(context)
        execution_results = [
            self.executors.execute(intent)
            for intent in plugin_batch.control_intents
        ]
        return RuntimeFrameResult(
            plugin_batch=plugin_batch,
            execution_results=execution_results,
        )

    def _active_model(self) -> dict | None:
        deployment = self.models.get_active_deployment()
        if deployment is None:
            return None

        project = self.models.get_project(deployment.project_id)
        artifact = self.models.get_artifact(deployment.artifact_id)
        return {
            "project": asdict(project) if project is not None else None,
            "deployment": asdict(deployment),
            "artifact": asdict(artifact) if artifact is not None else None,
        }
