"""Tests for the ROI handling the runtime service performs on every
captured frame.

The service contract is:
- inference receives the center-cropped ROI frame, not the raw source;
- a pre-cropped source ROI is reused without a second crop;
- the ROI metadata (offset, source dimensions) is exposed via
  `last_inference_status` for the UI to display.
"""
import numpy as np

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.inference import InferenceDetection, InferenceResult
from novasight.runtime import RuntimeService


class CapturingInference:
    def __init__(self) -> InferenceDetection | None:
        self.frame = None
        self._detection = InferenceDetection(0, 0.9, 10, 20, 30, 40)

    def infer(self, frame) -> InferenceResult:
        self.frame = frame
        return InferenceResult(
            available=True,
            detections=[self._detection],
            classes=["target"],
        )


def _make_frame(width: int, height: int) -> CapturedFrame:
    return CapturedFrame(
        frame_id=7,
        width=width,
        height=height,
        pixel_format="BGR",
        ts_ns=123,
        capture_wait_ms=1.5,
        image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def test_runtime_service_feeds_center_roi_to_inference() -> None:
    config = RuntimeConfig()
    config.roi.size = 320
    inference = CapturingInference()
    service = RuntimeService(
        config=config, models=None, executors=None, inference=inference,
    )

    service.process_captured_frame(_make_frame(1920, 1080))

    roi_frame = inference.frame
    assert roi_frame.width == 320
    assert roi_frame.height == 320
    assert roi_frame.source_width == 1920
    assert roi_frame.source_height == 1080
    assert roi_frame.offset_x == 800
    assert roi_frame.offset_y == 380
    assert roi_frame.image.shape == (320, 320, 3)

    status = service.last_inference_status
    assert status["input_width"] == 320
    assert status["input_height"] == 320
    assert status["source_width"] == 1920
    assert status["source_height"] == 1080
    assert status["roi_offset_x"] == 800
    assert status["roi_offset_y"] == 380
    assert status["detection_coordinate_space"] == "roi"


def test_runtime_service_reuses_capture_roi_without_second_crop() -> None:
    config = RuntimeConfig()
    config.roi.size = 320
    inference = CapturingInference()
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    frame = CapturedFrame(
        frame_id=8,
        width=320,
        height=320,
        pixel_format="BGR",
        ts_ns=123,
        capture_wait_ms=0.7,
        image=image,
        source_width=1920,
        source_height=1080,
        roi_size=320,
        roi_offset_x=800,
        roi_offset_y=380,
    )
    service = RuntimeService(
        config=config, models=None, executors=None, inference=inference,
    )

    service.process_captured_frame(frame)

    assert inference.frame.image is image
    assert inference.frame.source_width == 1920
    assert inference.frame.source_height == 1080
    assert inference.frame.offset_x == 800
    assert inference.frame.offset_y == 380
