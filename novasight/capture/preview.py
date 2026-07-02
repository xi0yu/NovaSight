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
    import cv2

    output = image.copy() if hasattr(image, "copy") else image
    center = (width // 2, height // 2)
    radius = max(4, int(min(width, height) * fov_ratio))
    cv2.circle(output, center, radius, (80, 220, 160), 2)
    for detection in detections:
        x1 = int(detection.x)
        y1 = int(detection.y)
        x2 = int(detection.x + detection.w)
        y2 = int(detection.y + detection.h)
        target = (int(detection.cx), int(detection.cy))
        cv2.rectangle(output, (x1, y1), (x2, y2), (80, 190, 255), 2)
        cv2.line(output, center, target, (180, 220, 120), 1)
        cv2.putText(
            output,
            f"{detection.cls}:{detection.score:.2f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 240, 210),
            1,
            cv2.LINE_AA,
        )
    return output
