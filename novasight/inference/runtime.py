from __future__ import annotations

from novasight.capture.source import CapturedFrame

from .contracts import InferenceEngine
from .contracts import InferenceResult
from .tensorrt import TensorRtInferenceEngine
from .unavailable import UnavailableInferenceEngine


class InferenceRuntime:
    def __init__(self, engine: InferenceEngine | None = None) -> None:
        self.engine = engine or TensorRtInferenceEngine()
        if not self.engine.available():
            self.engine = UnavailableInferenceEngine(
                self.engine.last_reason() or "TensorRT unavailable"
            )

    def status(self) -> dict:
        return self.engine.status()

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        try:
            result = self.engine.infer(frame)
        except Exception as exc:
            return InferenceResult(available=False, reason=str(exc))
        if not isinstance(result, InferenceResult):
            return InferenceResult(available=False, reason="invalid inference result")
        return result
