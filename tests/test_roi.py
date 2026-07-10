import numpy as np
import pytest

from novasight.capture.source import CapturedFrame, FrameResource
from novasight.contracts import Detection
from novasight.coordinates import CoordinateTransform
from novasight.roi import (
    ROI_SIZE_CHOICES,
    center_roi_frame,
    center_roi_region,
    map_detection_to_source,
)


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
    assert roi.capture_ts_ns == frame.capture_ts_ns
    assert roi.source_width == 1920
    assert roi.source_height == 1080
    assert roi.roi_size == 640
    assert roi.width == 640
    assert roi.height == 640
    assert roi.offset_x == 640
    assert roi.offset_y == 220
    assert roi.image.shape == (640, 640, 3)
    np.testing.assert_array_equal(roi.image, frame.image[220:860, 640:1280])


def test_center_roi_region_returns_source_center_crop() -> None:
    region = center_roi_region(source_width=1920, source_height=1080, requested_size=320)

    assert region == (800, 380, 320)


def test_coordinate_transform_round_trips_model_roi_capture_control_display() -> None:
    transform = CoordinateTransform(
        model_width=320,
        model_height=320,
        roi_x=640,
        roi_y=60,
        roi_width=960,
        roi_height=960,
        capture_width=1920,
        capture_height=1080,
        display_scale_x=0.5,
        display_scale_y=0.5,
    )

    roi_point = transform.model_to_roi_point(160, 160)
    capture_point = transform.roi_to_capture_point(roi_point.x, roi_point.y)
    control_point = transform.capture_to_control_point(capture_point.x, capture_point.y)
    display_point = transform.control_to_display_point(control_point.x, control_point.y)
    control_roundtrip = transform.display_to_control_point(display_point.x, display_point.y)
    capture_roundtrip = transform.control_to_capture_point(control_roundtrip.x, control_roundtrip.y)
    roi_roundtrip = transform.capture_to_roi_point(capture_roundtrip.x, capture_roundtrip.y)
    model_roundtrip = transform.roi_to_model_point(roi_roundtrip.x, roi_roundtrip.y)
    roi_box = Detection(cls=0, score=0.8, x=10, y=20, w=30, h=40).box
    control_box = transform.roi_to_control_box(roi_box)
    display_box = transform.roi_to_display_box(roi_box)

    assert roi_point.x == pytest.approx(480)
    assert roi_point.y == pytest.approx(480)
    assert capture_point.x == pytest.approx(1120)
    assert capture_point.y == pytest.approx(540)
    assert control_point.x == pytest.approx(1120)
    assert control_point.y == pytest.approx(540)
    assert control_box.x1 == pytest.approx(650)
    assert control_box.y1 == pytest.approx(80)
    assert control_box.x2 == pytest.approx(680)
    assert control_box.y2 == pytest.approx(120)
    assert display_box.x1 == pytest.approx(325)
    assert display_box.y1 == pytest.approx(40)
    assert display_box.x2 == pytest.approx(340)
    assert display_box.y2 == pytest.approx(60)
    assert model_roundtrip.x == pytest.approx(160, abs=1)
    assert model_roundtrip.y == pytest.approx(160, abs=1)


def test_center_roi_frame_reuses_pre_cropped_capture_roi() -> None:
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="BGR",
        source="appsink",
    )
    frame = CapturedFrame(
        frame_id=9,
        width=320,
        height=320,
        pixel_format="BGR",
        ts_ns=456,
        capture_wait_ms=0.8,
        image=image,
        frame_resource=resource,
        source_ts_ns=123_456,
        source_ts_kind="gstreamer_pts",
        source_width=1920,
        source_height=1080,
        roi_size=320,
        roi_offset_x=800,
        roi_offset_y=380,
    )

    roi = center_roi_frame(frame, requested_size=320)

    assert roi.source_width == 1920
    assert roi.source_height == 1080
    assert roi.roi_size == 320
    assert roi.offset_x == 800
    assert roi.offset_y == 380
    assert roi.image is image
    assert roi.frame_resource is resource
    assert roi.gpu_buffer is resource.handle
    assert roi.resource_memory == "nvmm"
    assert roi.receive_ts_ns == frame.receive_ts_ns
    assert roi.source_ts_ns == 123_456
    assert roi.source_ts_kind == "gstreamer_pts"


def test_center_roi_frame_reuses_pre_cropped_clamped_roi() -> None:
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    frame = CapturedFrame(
        frame_id=10,
        width=480,
        height=480,
        pixel_format="BGR",
        ts_ns=456,
        capture_wait_ms=0.8,
        image=image,
        source_width=640,
        source_height=480,
        roi_size=480,
        roi_offset_x=80,
        roi_offset_y=0,
    )

    roi = center_roi_frame(frame, requested_size=640)

    assert roi.source_width == 640
    assert roi.source_height == 480
    assert roi.roi_size == 480
    assert roi.offset_x == 80
    assert roi.offset_y == 0
    assert roi.image is image


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


def test_coordinate_transform_maps_roi_box_to_capture_space() -> None:
    transformer = CoordinateTransform(
        model_width=480,
        model_height=480,
        roi_x=720,
        roi_y=300,
        roi_width=480,
        roi_height=480,
        capture_width=1920,
        capture_height=1080,
    )
    detection = Detection(cls=1, score=0.9, x=10, y=20, w=30, h=40)

    mapped = transformer.roi_to_capture_box(detection.box)

    assert mapped.x1 == 730
    assert mapped.y1 == 320
    assert mapped.x2 == 760
    assert mapped.y2 == 360
