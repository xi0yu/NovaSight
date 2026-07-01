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
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlIntent:
    dx: float
    dy: float
    action: str | None
    confidence: float
    reason: str
    plugin_id: str


@dataclass(frozen=True)
class PluginBatchResult:
    plugin_results: list[PluginResult] = field(default_factory=list)
    control_intents: list[ControlIntent] = field(default_factory=list)


class VisionPlugin(Protocol):
    plugin_id: str

    def process(self, context: FrameContext) -> PluginResult:
        ...


class ControlPlugin(Protocol):
    plugin_id: str

    def process(self, context: FrameContext) -> ControlIntent | None:
        ...
