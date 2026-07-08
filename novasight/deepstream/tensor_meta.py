from __future__ import annotations

from typing import Any

from novasight.contracts import Detection, DetectionBatch
from novasight.coordinates import CoordinateTransform
from novasight.inference.postprocess.yolo import decode_nx6_detections
from novasight.model_registry.manifest import ModelManifest


def _normalize_output_tensor(output: Any, manifest: ModelManifest) -> Any:
    import numpy as np

    array = np.asarray(output)
    shape = [int(item) for item in array.shape]
    expected = [int(item) for item in manifest.output.shape]
    if not shape:
        raise ValueError("tensor shape is empty")
    if shape != expected:
        if len(expected) >= 2 and expected[0] == 1 and shape == expected[1:]:
            return array.reshape(tuple(expected))
        raise ValueError(f"tensor shape {shape} does not match manifest output shape {expected}")
    return array


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
    confidence_threshold: float | None = None,
    nms_threshold: float | None = None,
    metadata: dict[str, object] | None = None,
) -> DetectionBatch:
    _validate_manifest_contract(manifest)
    output = _normalize_output_tensor(output, manifest)
    debug: dict[str, Any] = {}
    model_detections = decode_nx6_detections(
        output,
        confidence_threshold=(
            manifest.postprocess.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        ),
        nms_threshold=(
            manifest.postprocess.nms_iou_threshold
            if nms_threshold is None
            else float(nms_threshold)
        ),
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
    classes = list(getattr(manifest.output, "class_names", []) or [])
    class_count = max(0, int(manifest.output.class_count))
    if len(classes) < class_count:
        classes.extend(str(index) for index in range(len(classes), class_count))
    return DetectionBatch(
        frame_id=int(frame_id),
        capture_ts_ns=int(capture_ts_ns),
        inference_start_ts_ns=int(inference_start_ts_ns),
        inference_end_ts_ns=int(inference_end_ts_ns),
        detections=roi_detections,
        classes=classes[:class_count],
        coordinate_space="roi",
        metadata=dict(metadata or {}),
    )


def _validate_manifest_contract(manifest: ModelManifest) -> None:
    parser = str(manifest.postprocess.parser or "").strip().lower()
    if parser != "yolo":
        raise ValueError(f"unsupported DeepStream postprocess parser: {manifest.postprocess.parser}")
    output_format = str(manifest.output.format or "").strip().lower()
    if output_format != "yolo_cxcywh_class_scores":
        raise ValueError(f"unsupported DeepStream output format: {manifest.output.format}")
    coordinate_mode = str(manifest.output.coordinate_mode or "").strip().lower()
    if coordinate_mode != "pixel":
        raise ValueError(f"unsupported DeepStream output coordinate mode: {manifest.output.coordinate_mode}")
    input_layout = str(manifest.input.layout or "").strip().upper()
    if input_layout != "NCHW" or len(manifest.input.shape) != 4:
        raise ValueError(
            f"DeepStream DetectionBatch mapping expects NCHW input shape, got {manifest.input.layout} {manifest.input.shape}"
        )
    class_count = int(manifest.output.class_count)
    expected_channels = 4 + class_count
    if class_count <= 0:
        raise ValueError(f"DeepStream YOLO output class_count must be positive, got {class_count}")
    if len(manifest.output.shape) != 3:
        raise ValueError(f"DeepStream YOLO output shape must be [batch, channels, candidates], got {manifest.output.shape}")
    actual_channels = int(manifest.output.shape[1])
    if actual_channels != expected_channels:
        raise ValueError(
            "DeepStream YOLO output channels must equal 4 + class_count "
            f"(channels={actual_channels}, class_count={class_count})"
        )
