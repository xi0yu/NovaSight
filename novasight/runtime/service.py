from __future__ import annotations

from dataclasses import asdict

from novasight.config import RuntimeConfig
from novasight.executors import ExecutorRegistry
from novasight.model_registry import ModelRegistry
from novasight.plugins import PluginRuntime

from .state import RuntimeState


class RuntimeService:
    def __init__(
        self,
        config: RuntimeConfig,
        models: ModelRegistry,
        plugins: PluginRuntime,
        executors: ExecutorRegistry,
    ) -> None:
        self.config = config
        self.models = models
        self.plugins = plugins
        self.executors = executors
        self.running = False

    def state(self) -> RuntimeState:
        return RuntimeState(
            running=self.running,
            source=self.config.source.default,
            active_model=self._active_model(),
            executor=self.executors.status(),
        )

    def _active_model(self) -> dict | None:
        deployments = self.models.list_deployments()
        if not deployments:
            return None

        deployment = deployments[0]
        project = self.models.get_project(deployment.project_id)
        artifact = self.models.get_artifact(deployment.artifact_id)
        return {
            "project": asdict(project) if project is not None else None,
            "deployment": asdict(deployment),
            "artifact": asdict(artifact) if artifact is not None else None,
        }
