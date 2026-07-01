import builtins
import sys
from pathlib import Path

import pytest

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.control import ControlOutput
from novasight.executors import ExecutorRegistry
from novasight.executors.dry_run import DryRunExecutor
from novasight.inference import (
    InferenceDetection,
    InferenceRuntime,
    InferenceResult,
    TensorRtInferenceEngine,
    UnavailableInferenceEngine,
)
from novasight.model_registry import ModelRegistry
from novasight.plugins import PluginRuntime
from novasight.runtime import RuntimeService


def test_unavailable_engine_reports_reason_and_returns_empty_result() -> None:
    engine = UnavailableInferenceEngine("TensorRT unavailable")

    assert engine.available() is False
    assert engine.status() == {
        "selected": "unavailable",
        "available": False,
        "loaded": False,
        "reason": "TensorRT unavailable",
    }

    result = engine.infer(
        CapturedFrame(1, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.available is False
    assert result.detections == []
    assert result.reason == "TensorRT unavailable"


def test_inference_result_detection_shape() -> None:
    result = InferenceResult(
        available=True,
        detections=[
            InferenceDetection(cls=0, score=0.9, x=10, y=20, w=30, h=40),
        ],
        classes=["target"],
    )

    assert result.detections[0].x == 10
    assert result.classes == ["target"]


def test_tensorrt_engine_requires_engine_artifact_suffix() -> None:
    engine = TensorRtInferenceEngine()

    with pytest.raises(ValueError, match="must be .engine"):
        engine.load(Path("model.onnx"), classes=["target"], input_shape="1x3x640x640")


def test_tensorrt_engine_reports_not_loaded_before_infer() -> None:
    engine = TensorRtInferenceEngine()

    result = engine.infer(
        CapturedFrame(1, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.available is False
    assert result.detections == []
    assert result.reason == "TensorRT engine not loaded"


def test_runtime_falls_back_to_unavailable_engine() -> None:
    class FakeUnavailableEngine:
        engine_id = "fake"

        def available(self) -> bool:
            return False

        def last_reason(self) -> str:
            return "fake unavailable"

        def status(self) -> dict:
            raise AssertionError("runtime should not call status while falling back")

        def load(
            self,
            artifact_path: Path,
            classes: list[str],
            input_shape: str,
        ) -> None:
            raise AssertionError("not used")

        def infer(self, frame: CapturedFrame) -> InferenceResult:
            raise AssertionError("not used")

    runtime = InferenceRuntime(FakeUnavailableEngine())

    assert isinstance(runtime.engine, UnavailableInferenceEngine)
    assert runtime.status() == {
        "selected": "unavailable",
        "available": False,
        "loaded": False,
        "reason": "fake unavailable",
    }


def test_runtime_probes_engine_once_and_preserves_first_unavailable_reason() -> None:
    class FakeUnavailableEngine:
        engine_id = "fake"

        def __init__(self) -> None:
            self.available_calls = 0
            self.status_calls = 0
            self.reason = ""

        def available(self) -> bool:
            self.available_calls += 1
            self.reason = "first unavailable reason"
            return False

        def last_reason(self) -> str:
            return self.reason

        def status(self) -> dict:
            self.status_calls += 1
            self.reason = "status changed reason"
            return {
                "selected": self.engine_id,
                "available": False,
                "loaded": False,
                "reason": self.reason,
            }

        def load(
            self,
            artifact_path: Path,
            classes: list[str],
            input_shape: str,
        ) -> None:
            raise AssertionError("not used")

        def infer(self, frame: CapturedFrame) -> InferenceResult:
            raise AssertionError("not used")

    engine = FakeUnavailableEngine()
    runtime = InferenceRuntime(engine)

    assert engine.available_calls == 1
    assert engine.status_calls == 0
    assert runtime.status()["reason"] == "first unavailable reason"


def test_missing_tensorrt_does_not_break_import_or_runtime_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tensorrt":
            raise ImportError("No module named tensorrt")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    runtime = InferenceRuntime(TensorRtInferenceEngine())
    status = runtime.status()

    assert status["selected"] == "unavailable"
    assert status["available"] is False
    assert status["loaded"] is False
    assert "TensorRT unavailable" in status["reason"]


class FakeInferenceRuntime:
    def status(self) -> dict:
        return {"selected": "fake", "available": True}

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        return InferenceResult(
            available=True,
            detections=[InferenceDetection(0, 0.9, 10, 20, 30, 40)],
            classes=["target"],
        )


def _runtime_service(tmp_path, inference) -> tuple[RuntimeService, DryRunExecutor]:
    dry_run = DryRunExecutor()
    service = RuntimeService(
        config=RuntimeConfig(),
        models=ModelRegistry(tmp_path / "db.sqlite", tmp_path / "models"),
        plugins=PluginRuntime.with_builtin_plugins(),
        executors=ExecutorRegistry([dry_run]),
        inference=inference,
    )
    return service, dry_run


def test_runtime_process_captured_frame_converts_inference_to_context(
    tmp_path,
) -> None:
    service, dry_run = _runtime_service(tmp_path, FakeInferenceRuntime())

    frame = CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    result = service.process_captured_frame(frame)

    assert result.plugin_batch.plugin_results
    assert len(result.plugin_batch.control_intents) == 1
    intent = result.plugin_batch.control_intents[0]
    assert intent.dx == -295
    assert intent.dy == 200
    assert intent.confidence == 0.9
    output = ControlOutput(
        dx=-120,
        dy=120,
        action="move",
        confidence=0.9,
        plugin_id="control.center_target",
        accepted=True,
        clipped=True,
        reason="clamped to configured limits",
    )
    assert [execution.intent for execution in result.execution_results] == [output]
    assert dry_run.history == [output]


def test_runtime_process_captured_frame_without_inference_uses_empty_context(
    tmp_path,
) -> None:
    service, dry_run = _runtime_service(tmp_path, None)

    result = service.process_captured_frame(
        CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.plugin_batch.plugin_results
    assert result.plugin_batch.control_intents == []
    assert result.execution_results == []
    assert dry_run.history == []


def test_runtime_process_captured_frame_unavailable_inference_uses_empty_context(
    tmp_path,
) -> None:
    class UnavailableRuntime:
        def infer(self, frame: CapturedFrame) -> InferenceResult:
            return InferenceResult(available=False, reason="not loaded")

    service, dry_run = _runtime_service(tmp_path, UnavailableRuntime())

    result = service.process_captured_frame(
        CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.plugin_batch.plugin_results
    assert result.plugin_batch.control_intents == []
    assert result.execution_results == []
    assert dry_run.history == []


def test_runtime_process_captured_frame_missing_infer_uses_empty_context(
    tmp_path,
) -> None:
    service, dry_run = _runtime_service(tmp_path, object())

    result = service.process_captured_frame(
        CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.plugin_batch.plugin_results
    assert result.plugin_batch.control_intents == []
    assert result.execution_results == []
    assert dry_run.history == []


def test_runtime_process_captured_frame_inference_exception_uses_empty_context(
    tmp_path,
) -> None:
    class BrokenRuntime:
        def infer(self, frame: CapturedFrame) -> InferenceResult:
            raise RuntimeError("inference failed")

    service, dry_run = _runtime_service(tmp_path, BrokenRuntime())

    result = service.process_captured_frame(
        CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.plugin_batch.plugin_results
    assert result.plugin_batch.control_intents == []
    assert result.execution_results == []
    assert dry_run.history == []


def test_runtime_process_captured_frame_none_result_uses_empty_context(
    tmp_path,
) -> None:
    class NoneRuntime:
        def infer(self, frame: CapturedFrame) -> None:
            return None

    service, dry_run = _runtime_service(tmp_path, NoneRuntime())

    result = service.process_captured_frame(
        CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.plugin_batch.plugin_results
    assert result.plugin_batch.control_intents == []
    assert result.execution_results == []
    assert dry_run.history == []


def test_runtime_process_captured_frame_result_without_available_uses_empty_context(
    tmp_path,
) -> None:
    class InvalidResultRuntime:
        def infer(self, frame: CapturedFrame) -> object:
            return object()

    service, dry_run = _runtime_service(tmp_path, InvalidResultRuntime())

    result = service.process_captured_frame(
        CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.plugin_batch.plugin_results
    assert result.plugin_batch.control_intents == []
    assert result.execution_results == []
    assert dry_run.history == []


def test_runtime_process_captured_frame_malformed_detection_uses_empty_context(
    tmp_path,
) -> None:
    class MalformedDetectionRuntime:
        def infer(self, frame: CapturedFrame) -> InferenceResult:
            return InferenceResult(available=True, detections=[object()])

    service, dry_run = _runtime_service(tmp_path, MalformedDetectionRuntime())

    result = service.process_captured_frame(
        CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    )

    assert result.plugin_batch.plugin_results
    assert result.plugin_batch.control_intents == []
    assert result.execution_results == []
    assert dry_run.history == []


def test_inference_runtime_infer_returns_unavailable_result_on_engine_exception() -> None:
    class BrokenEngine:
        engine_id = "broken"

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"selected": self.engine_id, "available": True}

        def load(
            self,
            artifact_path: Path,
            classes: list[str],
            input_shape: str,
        ) -> None:
            raise AssertionError("not used")

        def infer(self, frame: CapturedFrame) -> InferenceResult:
            raise RuntimeError("engine exploded")

    runtime = InferenceRuntime(BrokenEngine())

    result = runtime.infer(CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None))

    assert result.available is False
    assert result.detections == []
    assert result.reason == "engine exploded"


def test_inference_runtime_infer_normalizes_none_engine_result() -> None:
    class NoneEngine:
        engine_id = "none"

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"selected": self.engine_id, "available": True}

        def load(
            self,
            artifact_path: Path,
            classes: list[str],
            input_shape: str,
        ) -> None:
            raise AssertionError("not used")

        def infer(self, frame: CapturedFrame) -> None:
            return None

    runtime = InferenceRuntime(NoneEngine())

    result = runtime.infer(CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None))

    assert result.available is False
    assert result.detections == []
    assert result.reason == "invalid inference result"
