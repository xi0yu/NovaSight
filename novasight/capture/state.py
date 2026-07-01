from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CapturePreference = Literal[
    "auto_high_fps",
    "auto_low_latency",
    "auto_balanced",
    "manual",
]


@dataclass(frozen=True)
class CaptureCapability:
    pixel_format: str
    width: int
    height: int
    fps_list: list[int]


@dataclass(frozen=True)
class CaptureCapabilities:
    available: bool
    device: str
    capabilities: list[CaptureCapability] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class CaptureProfile:
    device: str
    pixel_format: str
    width: int
    height: int
    fps: int
    preference: CapturePreference
    selection_reason: str


@dataclass
class CaptureRuntimeState:
    available: bool = False
    device: str = "/dev/video0"
    profile: CaptureProfile | None = None
    backend: str | None = None
    fps_capture: float = 0.0
    frame_period_ms: float = 0.0
    capture_wait_ms: float = 0.0
    frames_dropped: int = 0
    recoveries: int = 0
    last_error: str | None = None
