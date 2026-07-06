"""Tests for the inference runtime and its contract with the runtime service.

The runtime is a defensive layer: it must convert broken engine responses,
exceptions, and missing inference into an empty FrameContext so that
downstream plugin control stays safe. The merged test covers that contract
once instead of repeating it across six near-identical tests.
"""
import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.executors import ExecutorRegistry
from novasight.control import ControlOutput
from novasight.executors.dry_run import DryRunExecutor
from novasight.inference import (
    InferenceDetection,
    InferenceResult,
    InferenceRuntime,
    OnnxRuntimeInferenceEngine,
    TensorRtInferenceEngine,
    UnavailableInferenceEngine,
)
from novasight.inference.input import parse_tensor_input_shape, prepare_tensor_input
from novasight.model_registry import ModelRegistry

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

    result = engine.infer(CapturedFrame(1, 640, 480, "BGR", 123, 1.0, image=None))

    assert result.available is False
    assert result.detections == []
    assert result.reason == "TensorRT unavailable"


def test_tensorrt_engine_rejects_non_engine_artifact() -> None:
    engine = TensorRtInferenceEngine()

    with pytest.raises(ValueError, match="must be .engine"):
        engine.load(Path("model.onnx"), classes=["target"], input_shape="1x3x640x640")


def test_tensorrt_engine_reports_not_loaded_before_infer() -> None:
    engine = TensorRtInferenceEngine()

    result = engine.infer(CapturedFrame(1, 640, 480, "BGR", 123, 1.0, image=None))

    assert result.available is False
    assert result.reason == "TensorRT engine not loaded"


def test_tensor_input_shape_parser_accepts_nchw_shapes() -> None:
    shape = parse_tensor_input_shape("1x3x640x640")

    assert shape.batch == 1
    assert shape.channels == 3
    assert shape.height == 640
    assert shape.width == 640


def test_tensor_input_preparer_prefers_gpu_buffer() -> None:
    frame = SimpleNamespace(
        frame_id=7,
        width=320,
        height=320,
        roi_size=320,
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=380,
        image=None,
        gpu_buffer=object(),
    )
    shape = parse_tensor_input_shape("1x3x640x640")

    prepared = prepare_tensor_input(frame, shape)

    assert prepared.mode == "gpu_buffer"
    assert prepared.width == 320
    assert prepared.height == 320
    assert prepared.needs_resize is True


def test_tensor_input_preparer_uses_cpu_image_fallback() -> None:
    frame = SimpleNamespace(
        frame_id=7,
        width=640,
        height=640,
        roi_size=640,
        source_width=1920,
        source_height=1080,
        offset_x=640,
        offset_y=220,
        image=object(),
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x640x640")

    prepared = prepare_tensor_input(frame, shape)

    assert prepared.mode == "cpu_image"
    assert prepared.needs_resize is False

def test_runtime_falls_back_to_unavailable_engine() -> None:
    class FakeUnavailableEngine:
        engine_id = "fake"

        def available(self) -> bool:
            return False

        def last_reason(self) -> str:
            return "fake unavailable"

        def status(self) -> dict:
            raise AssertionError("runtime should not call status while falling back")

        def load(self, artifact_path, classes, input_shape) -> None:
            raise AssertionError("not used")

        def infer(self, frame) -> InferenceResult:
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
                "selected": self.engine_id, "available": False,
                "loaded": False, "reason": self.reason,
            }

        def load(self, artifact_path, classes, input_shape) -> None:
            raise AssertionError("not used")

        def infer(self, frame) -> InferenceResult:
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

    def fake_import(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError("No module named tensorrt")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    status = InferenceRuntime(TensorRtInferenceEngine()).status()

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


def _runtime_service(tmp_path, inference):
    dry_run = DryRunExecutor()
    service = RuntimeService(
        config=RuntimeConfig(),
        models=ModelRegistry(tmp_path / "db.sqlite", tmp_path / "models"),
        executors=ExecutorRegistry([dry_run], default="dry_run"),
        inference=inference,
    )
    return service, dry_run

def test_runtime_service_publishes_inference_to_context_and_target(tmp_path) -> None:
    service, dry_run = _runtime_service(tmp_path, FakeInferenceRuntime())
    service.config.control.fov_ratio = 1.0
    frame = CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)

    service.process_captured_frame(frame)

    # The service must turn the inference result into a populated
    # FrameContext and a target selection; downstream strategy may
    # decide whether to emit an intent based on the hardware trigger.
    assert service.last_frame_context is not None
    assert service.last_frame_context.frame_id == 7
    assert service.last_frame_context.detections
    assert service.last_target is not None
    assert service.last_inference_status["available"] is True
    assert dry_run.history == []  # no trigger configured

def test_runtime_service_emits_clamped_intent_with_always_trigger(tmp_path) -> None:
    service, dry_run = _runtime_service(tmp_path, FakeInferenceRuntime())
    service.config.control.fov_ratio = 1.0
    service.config.control.trigger_mode = "always"
    frame = CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
    result = service.process_captured_frame(frame)

    assert len(result.control_intents) == 1
    assert len(result.execution_results) == 1
    output = result.execution_results[0].intent
    # The policy clamps the per-axis move to the configured limit; the
    # specific reason string is owned by ControlOutputPolicy.
    assert output.accepted is True
    assert output.clipped is True
    assert abs(output.dx) <= 120
    assert abs(output.dy) <= 120
    assert dry_run.history == [output]


def test_runtime_service_defensive_contract_for_broken_inference(tmp_path) -> None:
    """Any non-conforming inference input must keep the runtime silent.

    The runtime must never let a misbehaving inference (missing, broken,
    raising, returning None, or returning a malformed object) leak into
    the plugin pipeline. We verify the contract with one parametrized test
    rather than one test per failure mode.
    """
    scenarios: dict[str, object] = {
        "missing_inference": None,
        "broken_object": object(),
        "raising": type(
            "RaisingRuntime",
            (),
            {"infer": staticmethod(lambda frame: (_ for _ in ()).throw(RuntimeError("boom")))},
        )(),
        "none_result": type(
            "NoneResultRuntime",
            (),
            {"infer": staticmethod(lambda frame: None)},
        )(),
        "unavailable": type(
            "UnavailableRuntime",
            (),
            {"infer": staticmethod(lambda frame: InferenceResult(available=False, reason="nope"))},
        )(),
        "garbage_object": type(
            "GarbageRuntime",
            (),
            {"infer": staticmethod(lambda frame: object())},
        )(),
        "malformed_detection": type(
            "MalformedRuntime",
            (),
            {"infer": staticmethod(lambda frame: InferenceResult(available=True, detections=[object()]))},
        )(),
    }

    for name, inference in scenarios.items():
        service, dry_run = _runtime_service(tmp_path, inference)
        result = service.process_captured_frame(
            CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
        )
        assert result.control_intents == [], name
        assert result.execution_results == [], name
        assert dry_run.history == [], name
        assert service.last_inference_status["available"] is False, name


def test_runtime_infer_handles_engine_exceptions_and_invalid_results() -> None:
    class BrokenEngine:
        engine_id = "broken"

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"selected": self.engine_id, "available": True}

        def load(self, artifact_path, classes, input_shape) -> None:
            raise AssertionError("not used")

        def infer(self, frame) -> InferenceResult:
            raise RuntimeError("engine exploded")

    class NoneEngine:
        engine_id = "none"

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"selected": self.engine_id, "available": True}

        def load(self, artifact_path, classes, input_shape) -> None:
            raise AssertionError("not used")

        def infer(self, frame):
            return None

    broken_runtime = InferenceRuntime(BrokenEngine())
    none_runtime = InferenceRuntime(NoneEngine())

    broken_result = broken_runtime.infer(CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None))
    none_result = none_runtime.infer(CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None))

    assert broken_result.available is False
    assert broken_result.reason == "engine exploded"

    assert none_result.available is False
    assert none_result.reason == "invalid inference result"
