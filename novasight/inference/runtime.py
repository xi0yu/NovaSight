from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import InferenceEngine
from .contracts import InferenceResult
from .tensorrt import TensorRtInferenceEngine
from .unavailable import UnavailableInferenceEngine


class InferenceRuntime:
    def __init__(self, engine: InferenceEngine | None = None) -> None:
        self._load_error = ""
        self.engine = engine or TensorRtInferenceEngine()
        if not self.engine.available():
            self.engine = UnavailableInferenceEngine(
                self.engine.last_reason() or "TensorRT unavailable"
            )

    def status(self) -> dict:
        status = dict(self.engine.status())
        if self._load_error:
            status["available"] = False
            status["loaded"] = False
            status["reason"] = self._load_error
        return status

    def disable(self, reason: str) -> None:
        self._load_error = reason

    def load(
        self,
        artifact_path: Path,
        classes: list[str],
        input_shape: str,
    ) -> None:
        try:
            self.engine.load(artifact_path, classes, input_shape)
        except Exception as exc:
            self._load_error = str(exc)
            return
        self._load_error = ""

    def infer(self, frame: Any) -> InferenceResult:
        if self._load_error:
            return InferenceResult(available=False, reason=self._load_error)
        try:
            result = self.engine.infer(frame)
        except Exception as exc:
            return InferenceResult(available=False, reason=str(exc))
        if not isinstance(result, InferenceResult):
            return InferenceResult(available=False, reason="invalid inference result")
        return result
