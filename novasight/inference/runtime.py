from __future__ import annotations

from .contracts import InferenceEngine
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
