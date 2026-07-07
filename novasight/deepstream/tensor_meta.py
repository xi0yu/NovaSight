from __future__ import annotations

from typing import Any

from novasight.contracts import Detection, DetectionBatch
from novasight.coordinates import CoordinateTransform
from novasight.inference.postprocess.yolo import decode_nx6_detections
from novasight.model_registry.manifest import ModelManifest


def _validate_output_tensor_contract(output: Any, manifest: ModelManifest) -> None:
    import numpy as np

    array = np.asarray(output)
    shape = [int(item) for item in array.shape]
    expected = [int(item) for item in manifest.output.shape]
    if not shape:
        raise ValueError("tensor shape is empty")
    if len(shape) != len(expected):
        raise ValueError(f"tensor shape {shape} does not match manifest output shape {expected}")
    if shape != expected:
        raise ValueError(f"tensor shape {shape} does not match manifest output shape {expected}")


def output_tensor_to_detection_batch(
    output: Any,
    *,
    manifest: ModelManifest,
    frame_id: int,
    capture_ts_ns: int,
    inference_start_ts_ns: int,
    inference_end_ts_ns: int,
    roi_width: int,
    roi_height: int,
) -> DetectionBatch:
    _validate_output_tensor_contract(output, manifest)
    debug: dict[str, Any] = {}
    model_detections = decode_nx6_detections(
        output,
        confidence_threshold=manifest.postprocess.confidence_threshold,
        nms_threshold=manifest.postprocess.nms_iou_threshold,
        class_count=manifest.output.class_count,
        debug=debug,
    )
    transform = CoordinateTransform(
        model_width=int(manifest.input.shape[3]),
        model_height=int(manifest.input.shape[2]),
        roi_x=0,
        roi_y=0,
        roi_width=int(roi_width),
        roi_height=int(roi_height),
        capture_width=int(roi_width),
        capture_height=int(roi_height),
    )
    roi_detections = [
        Detection(
            cls=item.cls,
            score=item.score,
            box=transform.model_to_roi_box(item.box),
        )
        for item in model_detections
    ]
    return DetectionBatch(
        frame_id=int(frame_id),
        capture_ts_ns=int(capture_ts_ns),
        inference_start_ts_ns=int(inference_start_ts_ns),
        inference_end_ts_ns=int(inference_end_ts_ns),
        detections=roi_detections,
        classes=[str(index) for index in range(max(0, manifest.output.class_count))],
        coordinate_space="roi",
    )
