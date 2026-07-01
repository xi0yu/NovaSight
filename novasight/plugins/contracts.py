from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Detection:
    cls: int
    score: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class Track:
    track_id: int
    cls: int
    score: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class FrameContext:
    frame_id: int
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PluginResult:
    plugin_id: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlIntent:
    plugin_id: str
    dx: float
    dy: float
    target_id: int | None = None
    target_cls: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginBatchResult:
    vision_results: list[PluginResult] = field(default_factory=list)
    control_intents: list[ControlIntent] = field(default_factory=list)


class VisionPlugin(Protocol):
    plugin_id: str

    def process(self, context: FrameContext) -> PluginResult:
        ...


class ControlPlugin(Protocol):
    plugin_id: str

    def process(self, context: FrameContext) -> ControlIntent | None:
        ...
