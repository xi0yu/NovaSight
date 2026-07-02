from __future__ import annotations

from typing import Any

from novasight.capture.source import CapturedFrame
from novasight.plugins import Detection


def render_preview_frame(
    frame: CapturedFrame,
    *,
    runtime: Any | None = None,
    fov_ratio: float = 0.28,
) -> Any:
    image = frame.image
    if image is None:
        return image
    detections: list[Detection] = []
    try:
        inference = getattr(runtime, "inference", None) if runtime is not None else None
        if inference is not None and callable(getattr(inference, "infer", None)):
            inference_result = inference.infer(frame)
            detections = [
                Detection(
                    cls=item.cls,
                    score=item.score,
                    x=item.x,
                    y=item.y,
                    w=item.w,
                    h=item.h,
                )
                for item in getattr(inference_result, "detections", [])
            ]
    except Exception:
        detections = []
    return draw_overlay(image, width=frame.width, height=frame.height, detections=detections, fov_ratio=fov_ratio)


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
