from __future__ import annotations

from typing import Any

from novasight.coordinates import CoordinateTransform

from .contracts import InferenceDetection
from .input import PreparedTensorInput, TensorInputShape
from .preprocess import TensorPreprocessResult


def coordinate_transform_for_prepared_input(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    preprocess_result: TensorPreprocessResult | None = None,
) -> CoordinateTransform:
    metadata = dict(preprocess_result.metadata or {}) if preprocess_result is not None else {}
    model_content_width = _positive_float(metadata.get("model_content_width"), shape.width)
    model_content_height = _positive_float(metadata.get("model_content_height"), shape.height)
    pad_x = _non_negative_float(metadata.get("pad_x"))
    pad_y = _non_negative_float(metadata.get("pad_y"))
    return CoordinateTransform(
        model_width=model_content_width,
        model_height=model_content_height,
        roi_x=prepared.offset_x,
        roi_y=prepared.offset_y,
        roi_width=prepared.width,
        roi_height=prepared.height,
        capture_width=prepared.source_width,
        capture_height=prepared.source_height,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def map_model_detections_to_roi_frame(
    detections: list[InferenceDetection],
    *,
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    preprocess_result: TensorPreprocessResult | None = None,
) -> list[InferenceDetection]:
    if not detections:
        return []
    transform = coordinate_transform_for_prepared_input(
        prepared,
        shape,
        preprocess_result=preprocess_result,
    )
    return [
        InferenceDetection(
            cls=item.cls,
            score=item.score,
            box=transform.model_to_roi_box(item.box),
        )
        for item in detections
    ]


def preprocess_debug(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    preprocess_result: TensorPreprocessResult | None = None,
) -> dict[str, Any]:
    transform = coordinate_transform_for_prepared_input(
        prepared,
        shape,
        preprocess_result=preprocess_result,
    )
    roi_width = max(1, prepared.width)
    roi_height = max(1, prepared.height)
    roi_pixels = max(1, roi_width * roi_height)
    model_pixels = max(1, shape.width * shape.height)
    model_to_roi_scale_x = transform.model_to_roi_scale_x
    model_to_roi_scale_y = transform.model_to_roi_scale_y
    downscale_factor = max(model_to_roi_scale_x, model_to_roi_scale_y)
    payload = {
        "input_mode": prepared.mode,
        "resource_kind": prepared.resource_kind,
        "resource_memory": prepared.resource_memory,
        "pixel_format": prepared.pixel_format,
        "roi_width": roi_width,
        "roi_height": roi_height,
        "model_width": shape.width,
        "model_height": shape.height,
        "model_content_width": transform.model_width,
        "model_content_height": transform.model_height,
        "pad_x": transform.pad_x,
        "pad_y": transform.pad_y,
        "model_input_size": [shape.width, shape.height],
        "source_width": prepared.source_width,
        "source_height": prepared.source_height,
        "offset_x": prepared.offset_x,
        "offset_y": prepared.offset_y,
        "needs_resize": prepared.needs_resize,
        "model_to_roi_scale_x": model_to_roi_scale_x,
        "model_to_roi_scale_y": model_to_roi_scale_y,
        "roi_to_model_scale_x": transform.model_width / roi_width,
        "roi_to_model_scale_y": transform.model_height / roi_height,
        "downscale_factor": downscale_factor,
        "pixel_ratio": model_pixels / roi_pixels,
        "density_warning": prepared.needs_resize and downscale_factor > 1.25,
        "coordinate_transform": {
            "model_width": transform.model_width,
            "model_height": transform.model_height,
            "roi_x": transform.roi_x,
            "roi_y": transform.roi_y,
            "roi_width": transform.roi_width,
            "roi_height": transform.roi_height,
            "capture_width": transform.capture_width,
            "capture_height": transform.capture_height,
        },
    }
    if preprocess_result is not None:
        payload.update(preprocess_result.debug_payload())
    else:
        payload.update(
            {
                "preprocess_backend": "",
                "preprocess_zero_copy": False,
                "preprocess_reason": "",
            }
        )
    return payload


def _positive_float(value: object, default: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return parsed if parsed > 0.0 else float(default)


def _non_negative_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, parsed)
