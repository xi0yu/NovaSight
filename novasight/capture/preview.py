from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from novasight.capture.source import CapturedFrame, gstreamer_sample_to_bgr
from novasight.contracts import Detection
from novasight.roi import center_roi_frame

MAX_PREVIEW_DETECTION_FRAME_LAG = 6
logger = logging.getLogger("novasight.capture.preview")


def render_preview_frame(
    frame: CapturedFrame,
    *,
    runtime: Any | None = None,
    roi_size: int = 640,
) -> Any:
    preview_image = frame.image
    if preview_image is None:
        preview_image = getattr(frame, "preview_image", None)
    if preview_image is None:
        preview_resource = getattr(frame, "preview_resource", None)
        if getattr(preview_resource, "kind", "") == "gstreamer_preview_sample":
            try:
                preview_image = gstreamer_sample_to_bgr(preview_resource.handle)
            except Exception as exc:
                logger.warning("preview sample conversion failed: %s", exc)
    if preview_image is None:
        return None
    preview_frame = frame if preview_image is frame.image else replace(frame, image=preview_image)
    roi = center_roi_frame(
        preview_frame,
        requested_size=roi_size,
    )
    image = roi.image
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
    )


def draw_overlay(
    image: Any,
    *,
    width: int,
    height: int,
    detections: list[Detection],
) -> Any:
    from PIL import Image, ImageDraw

    output = _to_pil_rgb(image)
    if output is None:
        return image
    draw = ImageDraw.Draw(output)
    for detection in detections:
        x1 = int(detection.x)
        y1 = int(detection.y)
        x2 = int(detection.x + detection.w)
        y2 = int(detection.y + detection.h)
        draw.rectangle((x1, y1, x2, y2), outline=(80, 190, 255), width=2)
        draw.text((x1, max(2, y1 - 14)), f"{detection.cls}:{detection.score:.2f}", fill=(230, 240, 210))
    return output


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
        if frame_delta < 0 or frame_delta > MAX_PREVIEW_DETECTION_FRAME_LAG:
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
