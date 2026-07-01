import builtins
import sys
from pathlib import Path

import pytest

from novasight.capture.source import CapturedFrame
from novasight.inference import (
    InferenceDetection,
    InferenceRuntime,
    InferenceResult,
    TensorRtInferenceEngine,
    UnavailableInferenceEngine,
)


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
