import numpy as np
import pytest

from novasight.capture.source import CapturedFrame
from novasight.plugins import Detection
from novasight.roi import ROI_SIZE_CHOICES, center_roi_frame, map_detection_to_source


def make_frame(width: int, height: int) -> CapturedFrame:
    image = np.arange(height * width * 3, dtype=np.uint8).reshape((height, width, 3))
    return CapturedFrame(
        frame_id=7,
        width=width,
        height=height,
        pixel_format="BGR",
        ts_ns=123,
        capture_wait_ms=1.5,
        image=image,
    )


def test_allowed_roi_sizes_are_fixed() -> None:
    assert ROI_SIZE_CHOICES == (640, 480, 320, 256)


def test_center_roi_frame_crops_requested_square() -> None:
    frame = make_frame(width=1920, height=1080)

    roi = center_roi_frame(frame, requested_size=640)

    assert roi.frame_id == 7
    assert roi.source_width == 1920
    assert roi.source_height == 1080
    assert roi.roi_size == 640
    assert roi.width == 640
    assert roi.height == 640
    assert roi.offset_x == 640
    assert roi.offset_y == 220
    assert roi.image.shape == (640, 640, 3)
    np.testing.assert_array_equal(roi.image, frame.image[220:860, 640:1280])


def test_center_roi_frame_clamps_to_source_short_side() -> None:
    frame = make_frame(width=500, height=300)

    roi = center_roi_frame(frame, requested_size=640)

    assert roi.roi_size == 300
    assert roi.width == 300
    assert roi.height == 300
    assert roi.offset_x == 100
    assert roi.offset_y == 0
    assert roi.image.shape == (300, 300, 3)


def test_center_roi_frame_rejects_invalid_size() -> None:
    frame = make_frame(width=1920, height=1080)

    with pytest.raises(ValueError, match="unsupported ROI size"):
        center_roi_frame(frame, requested_size=512)


def test_map_detection_to_source_adds_roi_offset() -> None:
    detection = Detection(cls=0, score=0.8, x=10, y=20, w=30, h=40)

    mapped = map_detection_to_source(detection, offset_x=640, offset_y=220)

    assert mapped.cls == 0
    assert mapped.score == 0.8
    assert mapped.x == 650
    assert mapped.y == 240
    assert mapped.w == 30
    assert mapped.h == 40
