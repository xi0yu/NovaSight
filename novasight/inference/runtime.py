from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .contracts import InferenceEngine
from .contracts import InferenceResult
from .onnxruntime_engine import OnnxRuntimeInferenceEngine
from .tensorrt import TensorRtInferenceEngine
from .unavailable import UnavailableInferenceEngine


logger = logging.getLogger("novasight.inference.runtime")


class InferenceRuntime:
    def __init__(self, engine: InferenceEngine | None = None) -> None:
        self._load_error = ""
        self.confidence_threshold = 0.25
        self.nms_threshold = 0.45
        self._last_infer_error_logged = ""
        self.engine = engine or TensorRtInferenceEngine()
        if not self.engine.available():
            self.engine = UnavailableInferenceEngine(
                self.engine.last_reason() or "TensorRT unavailable"
            )

    def configure(
        self,
        *,
        confidence_threshold: float | None = None,
        nms_threshold: float | None = None,
    ) -> None:
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("nms_threshold", self.nms_threshold),
        ):
            if hasattr(self.engine, name):
                setattr(self.engine, name, value)

    def status(self) -> dict:
        status = dict(self.engine.status())
        if self._load_error:
            status["available"] = False
            status["loaded"] = False
            status["reason"] = self._load_error
        return status

    def disable(self, reason: str) -> None:
        self._load_error = reason
        logger.warning("inference disabled: %s", reason)

    def load(
        self,
        artifact_path: Path,
        classes: list[str],
        input_shape: str,
    ) -> None:
        suffix = artifact_path.suffix.lower()
        if suffix == ".onnx":
            self.engine = OnnxRuntimeInferenceEngine(
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
            )
        elif suffix == ".engine":
            self.engine = TensorRtInferenceEngine(
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
            )
        else:
            self.engine = UnavailableInferenceEngine(
                f"unsupported inference artifact suffix: {artifact_path.suffix}"
            )
        self.configure()
        try:
            self.engine.load(artifact_path, classes, input_shape)
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                "inference load failed artifact=%s engine=%s reason=%s",
                artifact_path,
                self.engine.engine_id,
                self._load_error,
            )
            return
        self._load_error = ""
        self._last_infer_error_logged = ""
        logger.info(
            "inference loaded artifact=%s engine=%s shape=%s classes=%d",
            artifact_path,
            self.engine.engine_id,
            input_shape,
            len(classes),
        )

    def infer(self, frame: Any) -> InferenceResult:
        if self._load_error:
            return InferenceResult(available=False, reason=self._load_error)
        try:
            result = self.engine.infer(frame)
        except Exception as exc:
            logger.exception("inference raised engine=%s", self.engine.engine_id)
            return InferenceResult(available=False, reason=str(exc))
        if not isinstance(result, InferenceResult):
            return InferenceResult(available=False, reason="invalid inference result")
        if not result.available and result.reason and result.reason != self._last_infer_error_logged:
            logger.warning(
                "inference failed engine=%s reason=%s",
                self.engine.engine_id,
                result.reason,
            )
            self._last_infer_error_logged = result.reason
        return result
