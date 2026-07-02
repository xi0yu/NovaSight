from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import InferenceResult


class TensorRtInferenceEngine:
    engine_id = "tensorrt"

    def __init__(self) -> None:
        self._reason = ""
        self._available = False
        self._loaded = False
        self._warmed = False

    def available(self) -> bool:
        try:
            import tensorrt  # noqa: F401
        except Exception as exc:
            self._reason = f"TensorRT unavailable: {exc}"
            self._available = False
            return False
        self._reason = ""
        self._available = True
        return True

    def last_reason(self) -> str:
        return self._reason

    def status(self) -> dict:
        return {
            "selected": self.engine_id,
            "available": self._available,
            "loaded": self._loaded,
            "warmed": self._warmed,
            "reason": self._reason,
        }

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        if not artifact_path.suffix == ".engine":
            raise ValueError(f"TensorRT artifact must be .engine: {artifact_path}")
        if not classes:
            raise ValueError("TensorRT classes must not be empty")
        if not input_shape.strip():
            raise ValueError("TensorRT input shape must not be empty")
        if not self.available():
            raise RuntimeError(self._reason)
        self._loaded = True
        self._warmup()

    def infer(self, frame: Any) -> InferenceResult:
        if not self._loaded:
            return InferenceResult(available=False, reason="TensorRT engine not loaded")
        return InferenceResult(available=True, detections=[], classes=[])

    def _warmup(self) -> None:
        self._warmed = True
