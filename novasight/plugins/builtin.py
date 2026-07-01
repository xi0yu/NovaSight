from __future__ import annotations

from novasight.plugins.contracts import (
    ControlIntent,
    Detection,
    FrameContext,
    PluginResult,
    Track,
)


class TrackStatsPlugin:
    plugin_id = "vision.track_stats"

    def process(self, context: FrameContext) -> PluginResult:
        return PluginResult(
            plugin_id=self.plugin_id,
            data={
                "detections": len(context.detections),
                "tracks": len(context.tracks),
            },
        )


class CenterTargetControlPlugin:
    plugin_id = "control.center_target"

    def process(self, context: FrameContext) -> ControlIntent | None:
        target = self._select_target(context)
        if target is None:
            return None

        return ControlIntent(
            plugin_id=self.plugin_id,
            dx=target.cx - context.width / 2,
            dy=context.height / 2 - target.cy,
            target_id=getattr(target, "track_id", None),
            target_cls=target.cls,
            metadata={"score": target.score},
        )

    def _select_target(self, context: FrameContext) -> Track | Detection | None:
        if context.tracks:
            return max(context.tracks, key=lambda item: item.score)
        if context.detections:
            return max(context.detections, key=lambda item: item.score)
        return None
