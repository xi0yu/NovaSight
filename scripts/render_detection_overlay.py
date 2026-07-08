#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from novasight.contracts import BBox, Detection, DetectionCoordinateSpace
from novasight.coordinates import CoordinateTransform


@dataclass(frozen=True)
class OverlayGeometry:
    image_width: int
    image_height: int
    roi_left: float
    roi_top: float
    roi_width: float
    roi_height: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Render DetectionBatch boxes over a captured frame.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--detections-json", required=True, help="DetectionBatch JSON or list of detections.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--roi-left", type=float, default=0.0)
    parser.add_argument("--roi-top", type=float, default=0.0)
    parser.add_argument("--roi-width", type=float, default=0.0)
    parser.add_argument("--roi-height", type=float, default=0.0)
    parser.add_argument(
        "--coordinate-space",
        default="",
        choices=["", "model", "roi", "capture", "control", "display"],
        help="Override DetectionBatch coordinate_space.",
    )
    args = parser.parse_args()

    render_overlay(
        image_path=Path(args.image),
        detections_path=Path(args.detections_json),
        output_path=Path(args.output),
        roi_left=args.roi_left,
        roi_top=args.roi_top,
        roi_width=args.roi_width,
        roi_height=args.roi_height,
        coordinate_space_override=args.coordinate_space or None,
    )
    print(f"wrote {args.output}")
    return 0


def render_overlay(
    *,
    image_path: Path,
    detections_path: Path,
    output_path: Path,
    roi_left: float = 0.0,
    roi_top: float = 0.0,
    roi_width: float = 0.0,
    roi_height: float = 0.0,
    coordinate_space_override: DetectionCoordinateSpace | None = None,
) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    payload = json.loads(detections_path.read_text(encoding="utf-8"))
    detections, coordinate_space = _detections_from_payload(payload)
    if coordinate_space_override is not None:
        coordinate_space = coordinate_space_override
    geometry = OverlayGeometry(
        image_width=int(image.width),
        image_height=int(image.height),
        roi_left=float(roi_left),
        roi_top=float(roi_top),
        roi_width=float(roi_width or image.width),
        roi_height=float(roi_height or image.height),
    )
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = _box_to_image_space(detection.box, coordinate_space=coordinate_space, geometry=geometry)
        color = _class_color(detection.cls)
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)
        label = _label(detection, payload)
        draw.rectangle((box.x1, max(0.0, box.y1 - 16.0), box.x1 + max(42, len(label) * 7), box.y1), fill=color)
        draw.text((box.x1 + 3, max(0.0, box.y1 - 15.0)), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _detections_from_payload(payload: Any) -> tuple[list[Detection], DetectionCoordinateSpace]:
    if isinstance(payload, dict):
        raw_detections = payload.get("detections", [])
        coordinate_space = str(payload.get("coordinate_space") or "roi")
    else:
        raw_detections = payload
        coordinate_space = "roi"
    if coordinate_space not in {"model", "roi", "capture", "control", "display"}:
        raise ValueError(f"unsupported coordinate_space: {coordinate_space}")
    if not isinstance(raw_detections, list):
        raise ValueError("detections payload must be a list or contain a detections list")
    return [_detection_from_mapping(item) for item in raw_detections], coordinate_space  # type: ignore[return-value]


def _detection_from_mapping(value: Any) -> Detection:
    if not isinstance(value, dict):
        raise ValueError("each detection must be an object")
    cls = int(value.get("cls", value.get("class_id", 0)))
    score = float(value.get("score", value.get("confidence", 0.0)))
    box = _box_from_mapping(value)
    return Detection(cls=cls, score=score, box=box)


def _box_from_mapping(value: dict[str, Any]) -> BBox:
    raw_box = value.get("box")
    if isinstance(raw_box, dict):
        return _box_from_mapping(raw_box)
    if isinstance(raw_box, list) and len(raw_box) == 4:
        return BBox.from_xyxy(float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
    if {"x1", "y1", "x2", "y2"}.issubset(value):
        return BBox.from_xyxy(value["x1"], value["y1"], value["x2"], value["y2"])
    if {"x", "y", "w", "h"}.issubset(value):
        return BBox.from_xywh(value["x"], value["y"], value["w"], value["h"])
    if {"left", "top", "width", "height"}.issubset(value):
        return BBox.from_xywh(value["left"], value["top"], value["width"], value["height"])
    raise ValueError(f"detection box fields missing: {sorted(value)}")


def _box_to_image_space(
    box: BBox,
    *,
    coordinate_space: DetectionCoordinateSpace,
    geometry: OverlayGeometry,
) -> BBox:
    transform = CoordinateTransform(
        model_width=geometry.roi_width,
        model_height=geometry.roi_height,
        roi_x=geometry.roi_left,
        roi_y=geometry.roi_top,
        roi_width=geometry.roi_width,
        roi_height=geometry.roi_height,
        capture_width=geometry.image_width,
        capture_height=geometry.image_height,
    )
    if coordinate_space == "roi":
        box = transform.roi_to_capture_box(box)
    elif coordinate_space == "model":
        box = transform.roi_to_capture_box(transform.model_to_roi_box(box))
    elif coordinate_space in {"capture", "control", "display"}:
        pass
    else:
        raise ValueError(f"unsupported coordinate_space: {coordinate_space}")
    return _clip_box(box, width=geometry.image_width, height=geometry.image_height)


def _clip_box(box: BBox, *, width: int, height: int) -> BBox:
    return BBox.from_xyxy(
        max(0.0, min(float(width), box.x1)),
        max(0.0, min(float(height), box.y1)),
        max(0.0, min(float(width), box.x2)),
        max(0.0, min(float(height), box.y2)),
    )


def _label(detection: Detection, payload: Any) -> str:
    classes = payload.get("classes", []) if isinstance(payload, dict) else []
    class_name = ""
    if isinstance(classes, list) and 0 <= int(detection.cls) < len(classes):
        class_name = str(classes[int(detection.cls)])
    prefix = class_name or str(detection.cls)
    return f"{prefix} {detection.score:.2f}"


def _class_color(cls: int) -> tuple[int, int, int]:
    colors = (
        (80, 190, 255),
        (255, 190, 70),
        (130, 230, 120),
        (255, 120, 160),
        (190, 150, 255),
    )
    return colors[int(cls) % len(colors)]


if __name__ == "__main__":
    raise SystemExit(main())
