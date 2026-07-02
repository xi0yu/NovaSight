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
    TensorRtInferenceEngine,
    UnavailableInferenceEngine,
)
from novasight.inference.input import parse_tensor_input_shape, prepare_tensor_input
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


def test_tensorrt_engine_records_selected_input_mode(monkeypatch) -> None:
    engine = TensorRtInferenceEngine()
    monkeypatch.setattr(engine, "available", lambda: True)
    frame = SimpleNamespace(
        frame_id=8,
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

    engine.load(Path("model.engine"), classes=["target"], input_shape="1x3x640x640")
    result = engine.infer(frame)

    assert result.available is True
    status = engine.status()
    assert status["input_shape"] == "1x3x640x640"
    assert status["last_input_mode"] == "gpu_buffer"
    assert status["last_input_needs_resize"] is True


def test_tensorrt_engine_rejects_empty_input_frame(monkeypatch) -> None:
    engine = TensorRtInferenceEngine()
    monkeypatch.setattr(engine, "available", lambda: True)
    frame = SimpleNamespace(
        frame_id=8,
        width=320,
        height=320,
        roi_size=320,
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=380,
        image=None,
        gpu_buffer=None,
    )

    engine.load(Path("model.engine"), classes=["target"], input_shape="1x3x640x640")
    result = engine.infer(frame)

    assert result.available is False
    assert result.reason == "TensorRT input frame has no gpu_buffer or image"


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


def test_runtime_load_failure_preserves_stable_unavailable_status() -> None:
    class FailingLoadEngine:
        engine_id = "fake"

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"selected": self.engine_id, "available": True}

        def load(self, artifact_path, classes, input_shape) -> None:
            raise RuntimeError("engine load failed")

        def infer(self, frame) -> InferenceResult:
            raise AssertionError("not used")

    runtime = InferenceRuntime(FailingLoadEngine())

    runtime.load(Path("model.engine"), ["target"], "1x3x640x640")

    assert runtime.status() == {
        "selected": "fake",
        "available": False,
        "loaded": False,
        "reason": "engine load failed",
    }


def test_runtime_can_recover_after_load_failure() -> None:
    class FlakyLoadEngine:
        engine_id = "fake"

        def __init__(self) -> None:
            self.fail = True
            self.loaded = False

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {
                "selected": self.engine_id, "available": True,
                "loaded": self.loaded, "reason": "",
            }

        def load(self, artifact_path, classes, input_shape) -> None:
            if self.fail:
                raise RuntimeError("engine load failed")
            self.loaded = True

        def infer(self, frame) -> InferenceResult:
            return InferenceResult(available=self.loaded)

    engine = FlakyLoadEngine()
    runtime = InferenceRuntime(engine)

    runtime.load(Path("model.engine"), ["target"], "1x3x640x640")
    assert runtime.status()["available"] is False
    engine.fail = False
    runtime.load(Path("model.engine"), ["target"], "1x3x640x640")
    status = runtime.status()
    assert status["available"] is True
    assert status["loaded"] is True
    assert status["reason"] == ""


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
        plugins=PluginRuntime.with_builtin_plugins(),
        executors=ExecutorRegistry([dry_run]),
        inference=inference,
    )
    return service, dry_run


def test_runtime_service_converts_inference_to_context_and_control(tmp_path) -> None:
    service, dry_run = _runtime_service(tmp_path, FakeInferenceRuntime())
    frame = CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)

    result = service.process_captured_frame(frame)

    assert result.plugin_batch.plugin_results
    assert len(result.plugin_batch.control_intents) == 1
    intent = result.plugin_batch.control_intents[0]
    assert intent.dx == -215
    assert intent.dy == 200
    assert intent.confidence == 0.9
    assert [execution.intent for execution in result.execution_results] == [
        ControlOutput(
            dx=-120, dy=120, action="move", confidence=0.9,
            plugin_id="control.center_target", accepted=True, clipped=True,
            reason="clamped to configured limits",
        )
    ]


def test_runtime_service_defensive_contract_for_broken_inference(tmp_path) -> None:
    """Any non-conforming inference input must produce an empty context.

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
        assert result.plugin_batch.plugin_results, name
        assert result.plugin_batch.control_intents == [], name
        assert result.execution_results == [], name
        assert dry_run.history == [], name


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
