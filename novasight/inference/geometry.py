from __future__ import annotations

from typing import Any

from novasight.coordinates import CoordinateTransform

from .contracts import InferenceDetection
from .input import PreparedTensorInput, TensorInputShape
from .preprocess import TensorPreprocessResult


def coordinate_transform_for_prepared_input(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
) -> CoordinateTransform:
    return CoordinateTransform(
        model_width=shape.width,
        model_height=shape.height,
        roi_x=prepared.offset_x,
        roi_y=prepared.offset_y,
        roi_width=prepared.width,
        roi_height=prepared.height,
        capture_width=prepared.source_width,
        capture_height=prepared.source_height,
    )


def map_model_detections_to_roi_frame(
    detections: list[InferenceDetection],
    *,
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
) -> list[InferenceDetection]:
    if not detections:
        return []
    transform = coordinate_transform_for_prepared_input(prepared, shape)
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
    transform = coordinate_transform_for_prepared_input(prepared, shape)
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
        "source_width": prepared.source_width,
        "source_height": prepared.source_height,
        "offset_x": prepared.offset_x,
        "offset_y": prepared.offset_y,
        "needs_resize": prepared.needs_resize,
        "model_to_roi_scale_x": model_to_roi_scale_x,
        "model_to_roi_scale_y": model_to_roi_scale_y,
        "roi_to_model_scale_x": shape.width / roi_width,
        "roi_to_model_scale_y": shape.height / roi_height,
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
