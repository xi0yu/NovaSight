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
    kind = "vision"

    def process(self, context: FrameContext) -> PluginResult:
        return PluginResult(
            plugin_id=self.plugin_id,
            kind=self.kind,
            payload={
                "detections": len(context.detections),
                "tracks": len(context.tracks),
            },
        )


class ExperimentalTargetPlugin:
    plugin_id = "vision.experimental_target"
    kind = "vision"

    def process(self, context: FrameContext) -> PluginResult:
        if context.width <= 0 or context.height <= 0:
            return PluginResult(
                self.plugin_id,
                self.kind,
                {"target": None, "reason": "invalid frame size"},
            )

        target = max(context.detections, key=lambda item: item.score, default=None)
        if target is None:
            return PluginResult(self.plugin_id, self.kind, {"target": None})

        cx = target.cx
        cy = target.cy
        class_name = (
            context.classes[target.cls]
            if 0 <= target.cls < len(context.classes)
            else str(target.cls)
        )
        return PluginResult(
            plugin_id=self.plugin_id,
            kind=self.kind,
            payload={
                "class_name": class_name,
                "score": target.score,
                "center": {"x": cx, "y": cy},
                "normalized_offset": {
                    "x": round((cx - context.width / 2) / (context.width / 2), 6),
                    "y": round((context.height / 2 - cy) / (context.height / 2), 6),
                },
                "reason": "highest-score detection",
            },
        )


class CenterTargetControlPlugin:
    plugin_id = "control.center_target"
    kind = "control"

    def process(self, context: FrameContext) -> ControlIntent | None:
        target = self._select_target(context)
        if target is None:
            return None

        return ControlIntent(
            dx=target.cx - context.width / 2,
            dy=context.height / 2 - target.cy,
            action="move",
            confidence=target.score,
            reason=self._reason(target),
            plugin_id=self.plugin_id,
        )

    def _select_target(self, context: FrameContext) -> Track | Detection | None:
        if context.tracks:
            return max(context.tracks, key=lambda item: item.score)
        if context.detections:
            return max(context.detections, key=lambda item: item.score)
        return None

    def _reason(self, target: Track | Detection) -> str:
        if isinstance(target, Track):
            return f"center highest-score track {target.track_id}"
        return "center highest-score detection"


class ExperimentalCenterControlPlugin:
    plugin_id = "control.experimental_center"
    kind = "control"

    def process(self, context: FrameContext) -> ControlIntent | None:
        if context.width <= 0 or context.height <= 0:
            return None

        target = max(context.detections, key=lambda item: item.score, default=None)
        if target is None:
            return None

        return ControlIntent(
            dx=target.cx - context.width / 2,
            dy=context.height / 2 - target.cy,
            action="move",
            confidence=target.score,
            reason="experimental center target",
            plugin_id=self.plugin_id,
        )
