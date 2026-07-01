from __future__ import annotations

from pathlib import Path

from novasight.capture.source import CapturedFrame

from .contracts import InferenceResult


class TensorRtInferenceEngine:
    engine_id = "tensorrt"

    def __init__(self) -> None:
        self._reason = ""
        self._available = False
        self._loaded = False

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
            "reason": self._reason,
        }

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        if not artifact_path.suffix == ".engine":
            raise ValueError(f"TensorRT artifact must be .engine: {artifact_path}")
        if not self.available():
            raise RuntimeError(self._reason)
        self._loaded = True

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        if not self._loaded:
            return InferenceResult(available=False, reason="TensorRT engine not loaded")
        return InferenceResult(available=True, detections=[], classes=[])
