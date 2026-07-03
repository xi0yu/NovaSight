from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from novasight.contracts import Detection


ROI_SIZE_CHOICES = (640, 480, 320, 256)


@dataclass(frozen=True)
class RoiFrame:
    frame_id: int
    source_width: int
    source_height: int
    roi_size: int
    offset_x: int
    offset_y: int
    ts_ns: int
    pixel_format: str
    image: Any | None
    gpu_buffer: Any | None = None

    @property
    def width(self) -> int:
        return self.roi_size

    @property
    def height(self) -> int:
        return self.roi_size


def normalize_roi_size(value: int) -> int:
    size = int(value)
    if size not in ROI_SIZE_CHOICES:
        raise ValueError(f"unsupported ROI size: {value}")
    return size


def center_roi_region(
    *,
    source_width: int,
    source_height: int,
    requested_size: int,
) -> tuple[int, int, int]:
    configured_size = normalize_roi_size(requested_size)
    roi_size = min(configured_size, int(source_width), int(source_height))
    offset_x = max(0, (int(source_width) - roi_size) // 2)
    offset_y = max(0, (int(source_height) - roi_size) // 2)
    return offset_x, offset_y, roi_size


def center_roi_frame(frame: Any, *, requested_size: int) -> RoiFrame:
    configured_size = normalize_roi_size(requested_size)
    frame_roi_size = getattr(frame, "roi_size", None)
    source_width = int(getattr(frame, "source_width", None) or frame.width)
    source_height = int(getattr(frame, "source_height", None) or frame.height)
    expected_offset_x, expected_offset_y, expected_roi_size = center_roi_region(
        source_width=source_width,
        source_height=source_height,
        requested_size=configured_size,
    )
    if (
        frame_roi_size == expected_roi_size
        and int(frame.width) == expected_roi_size
        and int(frame.height) == expected_roi_size
        and int(getattr(frame, "roi_offset_x", 0)) == expected_offset_x
        and int(getattr(frame, "roi_offset_y", 0)) == expected_offset_y
    ):
        return RoiFrame(
            frame_id=frame.frame_id,
            source_width=source_width,
            source_height=source_height,
            roi_size=expected_roi_size,
            offset_x=expected_offset_x,
            offset_y=expected_offset_y,
            ts_ns=frame.ts_ns,
            pixel_format=frame.pixel_format,
            image=frame.image,
            gpu_buffer=getattr(frame, "gpu_buffer", None),
        )

    offset_x, offset_y, roi_size = center_roi_region(
        source_width=int(frame.width),
        source_height=int(frame.height),
        requested_size=configured_size,
    )
    image = _crop_image(frame.image, offset_x=offset_x, offset_y=offset_y, size=roi_size)
    return RoiFrame(
        frame_id=frame.frame_id,
        source_width=frame.width,
        source_height=frame.height,
        roi_size=roi_size,
        offset_x=offset_x,
        offset_y=offset_y,
        ts_ns=frame.ts_ns,
        pixel_format=frame.pixel_format,
        image=image,
    )


def map_detection_to_source(
    detection: Detection,
    *,
    offset_x: int,
    offset_y: int,
) -> Detection:
    return Detection(
        cls=detection.cls,
        score=detection.score,
        x=detection.x + offset_x,
        y=detection.y + offset_y,
        w=detection.w,
        h=detection.h,
    )


def _crop_image(image: Any, *, offset_x: int, offset_y: int, size: int) -> Any | None:
    if image is None:
        return None
    if hasattr(image, "shape"):
        import numpy as np

        return np.ascontiguousarray(
            image[offset_y : offset_y + size, offset_x : offset_x + size]
        )
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.crop((offset_x, offset_y, offset_x + size, offset_y + size))
    except Exception:
        return None
    return None
