import numpy as np

from novasight.capture.preview import render_preview_frame
from novasight.capture.source import CapturedFrame


class RuntimeThatMustNotInfer:
    inference = object()


def test_render_preview_frame_crops_center_roi_without_inference() -> None:
    class ForbiddenInference:
        def infer(self, frame):
            raise AssertionError("preview must not call inference")

    runtime = RuntimeThatMustNotInfer()
    runtime.inference = ForbiddenInference()
    frame = CapturedFrame(
        frame_id=1,
        width=1920,
        height=1080,
        pixel_format="BGR",
        ts_ns=123,
        capture_wait_ms=1.0,
        image=np.zeros((1080, 1920, 3), dtype=np.uint8),
    )

    preview = render_preview_frame(frame, runtime=runtime, roi_size=320)

    assert preview.size == (320, 320)
