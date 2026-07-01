from __future__ import annotations

from collections.abc import Iterable

from novasight.plugins.builtin import CenterTargetControlPlugin, TrackStatsPlugin
from novasight.plugins.contracts import (
    ControlPlugin,
    FrameContext,
    PluginBatchResult,
    VisionPlugin,
)


class PluginRuntime:
    def __init__(
        self,
        vision_plugins: Iterable[VisionPlugin] = (),
        control_plugins: Iterable[ControlPlugin] = (),
    ) -> None:
        self.vision_plugins = list(vision_plugins)
        self.control_plugins = list(control_plugins)

    @classmethod
    def with_builtin_plugins(cls) -> PluginRuntime:
        return cls(
            vision_plugins=[TrackStatsPlugin()],
            control_plugins=[CenterTargetControlPlugin()],
        )

    def process(self, context: FrameContext) -> PluginBatchResult:
        vision_results = [
            plugin.process(context)
            for plugin in self.vision_plugins
        ]
        control_intents = [
            intent
            for plugin in self.control_plugins
            if (intent := plugin.process(context)) is not None
        ]
        return PluginBatchResult(
            vision_results=vision_results,
            control_intents=control_intents,
        )
