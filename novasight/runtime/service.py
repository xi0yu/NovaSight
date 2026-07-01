from __future__ import annotations

from dataclasses import asdict

from novasight.config import RuntimeConfig
from novasight.executors import ExecutorRegistry
from novasight.model_registry import ModelArtifact, ModelRegistry
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
        for project in self.models.list_projects():
            deployment = self.models.get_deployment(project.id)
            if deployment is None:
                continue
            artifact = self._find_artifact(deployment.artifact_id)
            return {
                "project": asdict(project),
                "deployment": asdict(deployment),
                "artifact": asdict(artifact) if artifact is not None else None,
            }
        return None

    def _find_artifact(self, artifact_id: int) -> ModelArtifact | None:
        for project in self.models.list_projects():
            for version in self.models.list_versions(project.id):
                for artifact in self.models.list_artifacts(version.id):
                    if artifact.id == artifact_id:
                        return artifact
        return None
