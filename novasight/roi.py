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
    offset_x: int = 0,
    offset_y: int = 0,
) -> tuple[int, int, int]:
    configured_size = normalize_roi_size(requested_size)
    roi_size = min(configured_size, int(source_width), int(source_height))
    base_x = (int(source_width) - roi_size) // 2
    base_y = (int(source_height) - roi_size) // 2
    roi_x = _clamp(base_x + int(offset_x), 0, int(source_width) - roi_size)
    roi_y = _clamp(base_y + int(offset_y), 0, int(source_height) - roi_size)
    return roi_x, roi_y, roi_size


def center_roi_frame(
    frame: Any,
    *,
    requested_size: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> RoiFrame:
    configured_size = normalize_roi_size(requested_size)
    frame_roi_size = getattr(frame, "roi_size", None)
    source_width = int(getattr(frame, "source_width", None) or frame.width)
    source_height = int(getattr(frame, "source_height", None) or frame.height)
    desired_offset_x, desired_offset_y, desired_roi_size = center_roi_region(
        source_width=source_width,
        source_height=source_height,
        requested_size=configured_size,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    image_width, image_height = _image_size(getattr(frame, "image", None))

    existing_offset_x = int(getattr(frame, "roi_offset_x", 0))
    existing_offset_y = int(getattr(frame, "roi_offset_y", 0))
    existing_roi_size = (
        int(frame_roi_size)
        if frame_roi_size is not None
        else int(frame.width) if int(frame.width) == int(frame.height) else None
    )
    frame_is_roi_view = (
        frame_roi_size is not None
        and existing_roi_size is not None
        and int(frame.width) == existing_roi_size
        and int(frame.height) == existing_roi_size
        and (image_width is None or image_width == existing_roi_size)
        and (image_height is None or image_height == existing_roi_size)
    )

    if frame_is_roi_view:
        if (
            existing_roi_size == desired_roi_size
            and existing_offset_x == desired_offset_x
            and existing_offset_y == desired_offset_y
        ):
            return RoiFrame(
                frame_id=frame.frame_id,
                source_width=source_width,
                source_height=source_height,
                roi_size=desired_roi_size,
                offset_x=desired_offset_x,
                offset_y=desired_offset_y,
                ts_ns=frame.ts_ns,
                pixel_format=frame.pixel_format,
                image=frame.image,
                gpu_buffer=getattr(frame, "gpu_buffer", None),
            )

        inner_x = desired_offset_x - existing_offset_x
        inner_y = desired_offset_y - existing_offset_y
        if (
            inner_x >= 0
            and inner_y >= 0
            and inner_x + desired_roi_size <= existing_roi_size
            and inner_y + desired_roi_size <= existing_roi_size
        ):
            return RoiFrame(
                frame_id=frame.frame_id,
                source_width=source_width,
                source_height=source_height,
                roi_size=desired_roi_size,
                offset_x=desired_offset_x,
                offset_y=desired_offset_y,
                ts_ns=frame.ts_ns,
                pixel_format=frame.pixel_format,
                image=_crop_image(
                    frame.image,
                    offset_x=inner_x,
                    offset_y=inner_y,
                    size=desired_roi_size,
                ),
                gpu_buffer=None,
            )

        return RoiFrame(
            frame_id=frame.frame_id,
            source_width=source_width,
            source_height=source_height,
            roi_size=existing_roi_size,
            offset_x=existing_offset_x,
            offset_y=existing_offset_y,
            ts_ns=frame.ts_ns,
            pixel_format=frame.pixel_format,
            image=frame.image,
            gpu_buffer=getattr(frame, "gpu_buffer", None),
        )

    if (
        frame_roi_size == desired_roi_size
        and int(frame.width) == desired_roi_size
        and int(frame.height) == desired_roi_size
        and int(getattr(frame, "roi_offset_x", 0)) == desired_offset_x
        and int(getattr(frame, "roi_offset_y", 0)) == desired_offset_y
    ):
        return RoiFrame(
            frame_id=frame.frame_id,
            source_width=source_width,
            source_height=source_height,
            roi_size=desired_roi_size,
            offset_x=desired_offset_x,
            offset_y=desired_offset_y,
            ts_ns=frame.ts_ns,
            pixel_format=frame.pixel_format,
            image=frame.image,
            gpu_buffer=getattr(frame, "gpu_buffer", None),
        )

    offset_x, offset_y, roi_size = center_roi_region(
        source_width=int(frame.width),
        source_height=int(frame.height),
        requested_size=configured_size,
        offset_x=offset_x,
        offset_y=offset_y,
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


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


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


def _image_size(image: Any) -> tuple[int | None, int | None]:
    if image is None:
        return None, None
    if hasattr(image, "shape"):
        try:
            return int(image.shape[1]), int(image.shape[0])
        except Exception:
            return None, None
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            return int(image.width), int(image.height)
    except Exception:
        return None, None
    return None, None
