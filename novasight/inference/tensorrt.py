from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import InferenceResult
from .input import PreparedTensorInput, TensorInputShape, parse_tensor_input_shape, prepare_tensor_input


class TensorRtInferenceEngine:
    engine_id = "tensorrt"

    def __init__(self) -> None:
        self._reason = ""
        self._available = False
        self._loaded = False
        self._warmed = False
        self._classes: list[str] = []
        self._input_shape: TensorInputShape | None = None
        self._last_input: PreparedTensorInput | None = None

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
            "supports_execution": False,
            "reason": self._reason,
            "input_shape": str(self._input_shape) if self._input_shape is not None else "",
            "last_input_mode": self._last_input.mode if self._last_input is not None else "",
            "last_input_needs_resize": (
                self._last_input.needs_resize if self._last_input is not None else False
            ),
        }

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        if not artifact_path.suffix == ".engine":
            raise ValueError(f"TensorRT artifact must be .engine: {artifact_path}")
        if not classes:
            raise ValueError("TensorRT classes must not be empty")
        if not input_shape.strip():
            raise ValueError("TensorRT input shape must not be empty")
        parsed_shape = parse_tensor_input_shape(input_shape)
        if not self.available():
            raise RuntimeError(self._reason)
        self._classes = list(classes)
        self._input_shape = parsed_shape
        self._last_input = None
        self._loaded = True
        self._warmup()

    def infer(self, frame: Any) -> InferenceResult:
        if not self._loaded:
            return InferenceResult(available=False, reason="TensorRT engine not loaded")
        if self._input_shape is None:
            return InferenceResult(available=False, reason="TensorRT input shape not loaded")
        try:
            self._last_input = prepare_tensor_input(frame, self._input_shape)
        except ValueError as exc:
            return InferenceResult(available=False, reason=str(exc))
        return InferenceResult(
            available=False,
            reason="TensorRT execution bindings are not implemented yet",
            classes=self._classes,
        )

    def _warmup(self) -> None:
        self._warmed = True
