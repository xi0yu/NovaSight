from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from novasight.plugins import Detection


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


def center_roi_frame(frame: Any, *, requested_size: int) -> RoiFrame:
    configured_size = normalize_roi_size(requested_size)
    roi_size = min(configured_size, int(frame.width), int(frame.height))
    offset_x = max(0, (int(frame.width) - roi_size) // 2)
    offset_y = max(0, (int(frame.height) - roi_size) // 2)
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
