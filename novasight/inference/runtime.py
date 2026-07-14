from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .contracts import InferenceEngine
from .contracts import InferenceResult
from .preprocess import GpuResourcePreprocessor
from .tensorrt import TensorRtInferenceEngine
from .unavailable import UnavailableInferenceEngine


logger = logging.getLogger("novasight.inference.runtime")


class InferenceRuntime:
    _UNSET = object()

    def __init__(
        self,
        engine: InferenceEngine | None = None,
        *,
        gpu_preprocessor: GpuResourcePreprocessor | None = None,
    ) -> None:
        self._load_error = ""
        self._last_switch_error = ""
        self.confidence_threshold = 0.25
        self.nms_threshold = 0.45
        self._last_infer_error_logged = ""
        self._gpu_preprocessor = gpu_preprocessor
        self._engine_lock = threading.RLock()
        self._prepare_lock = threading.Lock()
        self.engine = engine or TensorRtInferenceEngine()
        if not self.engine.available():
            self.engine = UnavailableInferenceEngine(
                self.engine.last_reason() or "TensorRT unavailable"
            )
        self._configure_engine_gpu_preprocessor()

    def configure(
        self,
        *,
        confidence_threshold: float | None = None,
        nms_threshold: float | None = None,
        gpu_preprocessor: GpuResourcePreprocessor | None | object = _UNSET,
    ) -> None:
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
        if gpu_preprocessor is not self._UNSET:
            self._gpu_preprocessor = gpu_preprocessor  # type: ignore[assignment]
        with self._engine_lock:
            for name, value in (
                ("confidence_threshold", self.confidence_threshold),
                ("nms_threshold", self.nms_threshold),
            ):
                if hasattr(self.engine, name):
                    setattr(self.engine, name, value)
            setter = getattr(self.engine, "set_gpu_preprocessor", None)
            if callable(setter):
                setter(self._gpu_preprocessor)

    def status(self) -> dict:
        with self._engine_lock:
            status = dict(self.engine.status())
        status["gpu_preprocessor"] = self._gpu_preprocessor_status()
        if self._load_error:
            status["available"] = False
            status["loaded"] = False
            status["reason"] = self._load_error
        if self._last_switch_error:
            status["last_switch_error"] = self._last_switch_error
        return status

    def _gpu_preprocessor_status(self) -> dict[str, Any]:
        if self._gpu_preprocessor is None:
            return {
                "selected": "",
                "enabled": False,
                "available": False,
                "reason": "disabled",
            }
        status = getattr(self._gpu_preprocessor, "status", None)
        if callable(status):
            return dict(status())
        return {
            "selected": type(self._gpu_preprocessor).__name__,
            "enabled": True,
            "available": True,
            "reason": "",
        }

    def _configure_engine_gpu_preprocessor(self) -> None:
        setter = getattr(self.engine, "set_gpu_preprocessor", None)
        if callable(setter):
            setter(self._gpu_preprocessor)

    def disable(self, reason: str) -> None:
        self._load_error = reason
        logger.warning("inference disabled: %s", reason)

    def unload(self, reason: str) -> None:
        replacement = UnavailableInferenceEngine(reason)
        with self._engine_lock:
            previous = self.engine
            self.engine = replacement
            self._load_error = reason
            self._last_switch_error = ""
            close = getattr(previous, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        logger.info("inference engine unloaded: %s", reason)

    def record_switch_error(self, reason: str) -> None:
        self._last_switch_error = reason
        logger.warning("inference model switch failed: %s", reason)

    def probe(
        self,
        artifact_path: Path,
        classes: list[str],
        input_shape: str,
    ) -> dict:
        candidate: InferenceEngine | None = None
        try:
            candidate, status = self.prepare(artifact_path, classes, input_shape)
            return status
        except Exception as exc:
            return {
                "selected": getattr(candidate, "engine_id", "unknown"),
                "available": False,
                "loaded": False,
                "reason": str(exc),
            }
        finally:
            close = getattr(candidate, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def load(
        self,
        artifact_path: Path,
        classes: list[str],
        input_shape: str,
    ) -> None:
        try:
            candidate, _status = self.prepare(artifact_path, classes, input_shape)
        except Exception as exc:
            self._last_switch_error = str(exc)
            logger.warning(
                "inference load failed artifact=%s reason=%s",
                artifact_path,
                self._last_switch_error,
            )
            return
        self.commit(candidate, artifact_path=artifact_path, classes=classes, input_shape=input_shape)

    def prepare(
        self,
        artifact_path: Path,
        classes: list[str],
        input_shape: str,
    ) -> tuple[InferenceEngine, dict]:
        candidate = self._engine_for_artifact(artifact_path)
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("nms_threshold", self.nms_threshold),
        ):
            if hasattr(candidate, name):
                setattr(candidate, name, value)
        try:
            with self._prepare_lock:
                candidate.load(artifact_path, classes, input_shape)
                status = dict(candidate.status())
            status["loaded"] = status.get("loaded") is True
            return candidate, status
        except Exception:
            close = getattr(candidate, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise

    def prepare_profile(
        self,
        profile: Any,
        *,
        diagnostic: bool = False,
    ) -> tuple[InferenceEngine, dict]:
        artifact_path = Path(profile.engine.path).expanduser().resolve(strict=False)
        candidate = self._engine_for_artifact(artifact_path)
        for name, value in (
            ("confidence_threshold", float(profile.postprocess.confidence_threshold)),
            ("nms_threshold", float(profile.postprocess.nms_threshold)),
        ):
            if hasattr(candidate, name):
                setattr(candidate, name, value)
        configure_profile = getattr(candidate, "configure_model_profile", None)
        if not callable(configure_profile):
            raise RuntimeError(
                f"inference engine {candidate.engine_id} does not support ModelProfile"
            )
        set_diagnostic_mode = getattr(candidate, "set_diagnostic_mode", None)
        try:
            configure_profile(profile)
            if callable(set_diagnostic_mode):
                set_diagnostic_mode(bool(diagnostic))
            input_shape = "x".join(str(value) for value in profile.input.runtime_shape)
            with self._prepare_lock:
                candidate.load(artifact_path, list(profile.labels), input_shape)
                status = dict(candidate.status())
            status["loaded"] = status.get("loaded") is True
            status["diagnostic"] = bool(diagnostic)
            status["model_id"] = str(profile.model_id)
            return candidate, status
        except Exception:
            close = getattr(candidate, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise

    def commit(
        self,
        candidate: InferenceEngine,
        *,
        artifact_path: Path,
        classes: list[str],
        input_shape: str,
    ) -> None:
        with self._engine_lock:
            previous_engine = self.engine
            self.engine = candidate
            runtime_status = dict(self.engine.status())
            self._load_error = ""
            self._last_switch_error = ""
            self._last_infer_error_logged = ""
            close = getattr(previous_engine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        logger.info(
            "inference loaded artifact=%s engine=%s registered_shape=%s runtime_shape=%s shape_source=%s classes=%d",
            artifact_path,
            self.engine.engine_id,
            input_shape,
            runtime_status.get("input_shape", ""),
            runtime_status.get("input_shape_source", ""),
            len(classes),
        )

    def _engine_for_artifact(self, artifact_path: Path) -> InferenceEngine:
        suffix = artifact_path.suffix.lower()
        if suffix == ".engine":
            return TensorRtInferenceEngine(
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                gpu_preprocessor=self._gpu_preprocessor,
            )
        return UnavailableInferenceEngine(
            f"unsupported inference artifact suffix: {artifact_path.suffix}"
        )

    def infer(self, frame: Any) -> InferenceResult:
        if self._load_error:
            return InferenceResult(available=False, reason=self._load_error)
        try:
            with self._engine_lock:
                engine = self.engine
                result = engine.infer(frame)
        except Exception as exc:
            engine_id = getattr(locals().get("engine", None), "engine_id", "unknown")
            logger.exception("inference raised engine=%s", engine_id)
            return InferenceResult(available=False, reason=str(exc))
        if not isinstance(result, InferenceResult):
            return InferenceResult(available=False, reason="invalid inference result")
        if not result.available and result.reason and result.reason != self._last_infer_error_logged:
            logger.warning(
                "inference failed engine=%s reason=%s",
                engine.engine_id,
                result.reason,
            )
            self._last_infer_error_logged = result.reason
        return result
