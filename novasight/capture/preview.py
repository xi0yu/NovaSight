from __future__ import annotations

from typing import Any

from novasight.capture.source import CapturedFrame
from novasight.contracts import Detection
from novasight.roi import center_roi_frame


def render_preview_frame(
    frame: CapturedFrame,
    *,
    runtime: Any | None = None,
    roi_size: int = 640,
    roi_offset_x: int = 0,
    roi_offset_y: int = 0,
    fov_ratio: float = 0.28,
) -> Any:
    roi = center_roi_frame(
        frame,
        requested_size=roi_size,
        offset_x=roi_offset_x,
        offset_y=roi_offset_y,
    )
    image = roi.image
    if image is None:
        return image
    fov_ratio = _runtime_fov_ratio(runtime, fallback=fov_ratio)
    detections = _runtime_roi_detections(
        runtime,
        frame_id=frame.frame_id,
        offset_x=roi.offset_x,
        offset_y=roi.offset_y,
        roi_size=roi.width,
    )
    return draw_overlay(
        image,
        width=roi.width,
        height=roi.height,
        detections=detections,
        fov_ratio=fov_ratio,
    )


def draw_overlay(
    image: Any,
    *,
    width: int,
    height: int,
    detections: list[Detection],
    fov_ratio: float = 0.28,
) -> Any:
    from PIL import Image, ImageDraw

    output = _to_pil_rgb(image)
    if output is None:
        return image
    draw = ImageDraw.Draw(output)
    center = (width // 2, height // 2)
    radius = max(4, int(min(width, height) * fov_ratio))
    draw.ellipse(
        (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
        outline=(80, 220, 160),
        width=2,
    )
    for detection in detections:
        x1 = int(detection.x)
        y1 = int(detection.y)
        x2 = int(detection.x + detection.w)
        y2 = int(detection.y + detection.h)
        target = (int(detection.cx), int(detection.cy))
        draw.rectangle((x1, y1, x2, y2), outline=(80, 190, 255), width=2)
        draw.line((center, target), fill=(180, 220, 120), width=1)
        draw.text((x1, max(2, y1 - 14)), f"{detection.cls}:{detection.score:.2f}", fill=(230, 240, 210))
    return output


def _runtime_fov_ratio(runtime: Any | None, *, fallback: float) -> float:
    config = getattr(runtime, "config", None)
    control = getattr(config, "control", None)
    value = getattr(control, "fov_ratio", fallback)
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return fallback
    if ratio <= 0 or ratio > 1:
        return fallback
    return ratio


def _runtime_roi_detections(
    runtime: Any | None,
    *,
    frame_id: int,
    offset_x: int,
    offset_y: int,
    roi_size: int,
) -> list[Detection]:
    context = getattr(runtime, "last_frame_context", None)
    if context is None:
        return []
    context_frame_id = getattr(context, "frame_id", None)
    if context_frame_id != frame_id:
        try:
            frame_delta = int(frame_id) - int(context_frame_id)
        except (TypeError, ValueError):
            return []
        if frame_delta < 0 or frame_delta > 240:
            return []
    detections: list[Detection] = []
    context_width = int(getattr(context, "width", 0) or 0)
    context_height = int(getattr(context, "height", 0) or 0)
    uses_roi_coordinates = context_width == roi_size and context_height == roi_size
    for detection in getattr(context, "detections", []):
        if uses_roi_coordinates:
            x = float(detection.x)
            y = float(detection.y)
        else:
            x = float(detection.x) - offset_x
            y = float(detection.y) - offset_y
        w = float(detection.w)
        h = float(detection.h)
        if x + w < 0 or y + h < 0 or x > roi_size or y > roi_size:
            continue
        detections.append(
            Detection(
                cls=int(detection.cls),
                score=float(detection.score),
                x=max(0.0, x),
                y=max(0.0, y),
                w=min(w, roi_size - max(0.0, x)),
                h=min(h, roi_size - max(0.0, y)),
            )
        )
    return detections


def _to_pil_rgb(image: Any):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    if hasattr(image, "shape"):
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
            return Image.fromarray(rgb, mode="RGB")
    return None
