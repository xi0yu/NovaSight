from novasight.capture.source import CapturedFrame
from novasight.inference import (
    InferenceDetection,
    InferenceResult,
    UnavailableInferenceEngine,
)


def test_unavailable_engine_reports_reason_and_returns_empty_result() -> None:
    engine = UnavailableInferenceEngine("TensorRT unavailable")

    assert engine.available() is False
    assert engine.status()["reason"] == "TensorRT unavailable"

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
