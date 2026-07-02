from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class InferenceDetection:
    cls: int
    score: float
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class InferenceResult:
    available: bool
    detections: list[InferenceDetection] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    reason: str = ""


class InferenceEngine(Protocol):
    engine_id: str

    def available(self) -> bool: ...
    def last_reason(self) -> str: ...
    def status(self) -> dict: ...
    def load(
        self,
        artifact_path: Path,
        classes: list[str],
        input_shape: str,
    ) -> None: ...
    def infer(self, frame: Any) -> InferenceResult: ...
