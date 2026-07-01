from __future__ import annotations

from pathlib import Path

from novasight.capture.source import CapturedFrame

from .contracts import InferenceResult


class UnavailableInferenceEngine:
    engine_id = "unavailable"

    def __init__(self, reason: str = "inference engine unavailable") -> None:
        self.reason = reason

    def available(self) -> bool:
        return False

    def last_reason(self) -> str:
        return self.reason

    def status(self) -> dict:
        return {
            "selected": self.engine_id,
            "available": False,
            "loaded": False,
            "reason": self.reason,
        }

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        raise RuntimeError(self.reason)

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        return InferenceResult(available=False, reason=self.reason)
