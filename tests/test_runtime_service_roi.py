import numpy as np

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.inference import InferenceDetection, InferenceResult
from novasight.plugins import FrameContext, PluginBatchResult
from novasight.runtime import RuntimeService


class CapturingInference:
    def __init__(self) -> None:
        self.frame = None

    def infer(self, frame) -> InferenceResult:
        self.frame = frame
        return InferenceResult(
            available=True,
            detections=[InferenceDetection(0, 0.9, 10, 20, 30, 40)],
            classes=["target"],
        )


class CapturingPlugins:
    def __init__(self) -> None:
        self.context: FrameContext | None = None

    def process(self, context: FrameContext) -> PluginBatchResult:
        self.context = context
        return PluginBatchResult()


class NoopExecutors:
    def status(self) -> dict:
        return {}

    def execute(self, intent):
        raise AssertionError("no control intent should be emitted")


def test_runtime_service_feeds_roi_to_inference_and_maps_detections_to_source() -> None:
    config = RuntimeConfig()
    config.roi.size = 320
    inference = CapturingInference()
    plugins = CapturingPlugins()
    frame = CapturedFrame(
        frame_id=7,
        width=1920,
        height=1080,
        pixel_format="BGR",
        ts_ns=123,
        capture_wait_ms=1.5,
        image=np.zeros((1080, 1920, 3), dtype=np.uint8),
    )
    service = RuntimeService(
        config=config,
        models=None,
        plugins=plugins,
        executors=NoopExecutors(),
        inference=inference,
    )

    service.process_captured_frame(frame)

    assert inference.frame is not None
    assert inference.frame.width == 320
    assert inference.frame.height == 320
    assert inference.frame.source_width == 1920
    assert inference.frame.source_height == 1080
    assert inference.frame.offset_x == 800
    assert inference.frame.offset_y == 380
    assert inference.frame.image.shape == (320, 320, 3)

    assert plugins.context is not None
    assert plugins.context.frame_id == 7
    assert plugins.context.width == 1920
    assert plugins.context.height == 1080
    assert plugins.context.classes == ["target"]
    assert plugins.context.detections[0].x == 810
    assert plugins.context.detections[0].y == 400
    assert plugins.context.detections[0].w == 30
    assert plugins.context.detections[0].h == 40


def test_runtime_service_reuses_capture_roi_without_second_crop() -> None:
    config = RuntimeConfig()
    config.roi.size = 320
    inference = CapturingInference()
    plugins = CapturingPlugins()
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
        config=config,
        models=None,
        plugins=plugins,
        executors=NoopExecutors(),
        inference=inference,
    )

    service.process_captured_frame(frame)

    assert inference.frame is not None
    assert inference.frame.image is image
    assert inference.frame.source_width == 1920
    assert inference.frame.source_height == 1080
    assert inference.frame.offset_x == 800
    assert inference.frame.offset_y == 380
    assert plugins.context is not None
    assert plugins.context.width == 1920
    assert plugins.context.height == 1080
    assert plugins.context.detections[0].x == 810
    assert plugins.context.detections[0].y == 400
