from __future__ import annotations

from typing import Any

from novasight.inference.contracts import InferenceDetection


MAX_NMS_CANDIDATES = 300


def decode_nx6_detections(
    output: Any,
    *,
    confidence_threshold: float,
    nms_threshold: float,
    class_count: int | None = None,
    debug: dict[str, Any] | None = None,
) -> list[InferenceDetection]:
    import numpy as np

    array = np.asarray(output, dtype=np.float32)
    original_shape = tuple(int(item) for item in getattr(array, "shape", ()))
    if array.size == 0:
        _update_decode_debug(debug, output_shape=original_shape, reason="empty output")
        return []
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        _update_decode_debug(
            debug,
            output_shape=original_shape,
            squeezed_shape=tuple(int(item) for item in getattr(array, "shape", ())),
            reason="unsupported output dimensions",
        )
        return []

    transposed_from_channel_first = 4 < array.shape[0] < array.shape[1]
    candidates_first = array.T if transposed_from_channel_first else array
    if candidates_first.shape[1] < 5:
        _update_decode_debug(
            debug,
            output_shape=original_shape,
            squeezed_shape=tuple(int(item) for item in array.shape),
            candidates_shape=tuple(int(item) for item in candidates_first.shape),
            reason="fewer than 5 prediction columns",
        )
        return []

    expected_classes = max(0, int(class_count or 0))
    columns = int(candidates_first.shape[1])
    layout_modes = _select_decode_layouts(
        columns=columns,
        class_count=expected_classes,
        transposed_from_channel_first=transposed_from_channel_first,
    )
    variants: list[tuple[list[InferenceDetection], dict[str, Any]]] = []
    for layout in layout_modes:
        if layout in {"yolov8-cxcywh-cls", "yolov5-cxcywh-obj-cls"}:
            variants.append(
                _decode_yolo_scores(
                    candidates_first,
                    mode=layout,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                )
            )
        elif layout == "xyxy-score-cls":
            variants.append(
                _decode_xyxy_score_cls(
                    candidates_first,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                )
            )
    if not variants:
        _update_decode_debug(
            debug,
            output_shape=original_shape,
            squeezed_shape=tuple(int(item) for item in array.shape),
            candidates_shape=tuple(int(item) for item in candidates_first.shape),
            expected_classes=expected_classes,
            prediction_columns=columns,
            reason="no supported decode layout",
        )
        return []
    for _detections, stats in variants:
        stats["shape_match"] = _layout_shape_matches(
            str(stats.get("layout", "")),
            columns=columns,
            class_count=expected_classes,
        )
        stats["orientation_match"] = _layout_orientation_matches(
            str(stats.get("layout", "")),
            transposed_from_channel_first=transposed_from_channel_first,
        )
    best_detections, best_stats = max(
        variants,
        key=lambda item: (
            int(
                bool(item[1].get("class_id_like", False))
                and int(item[1].get("nms_detections", 0)) > 0
            ),
            int(bool(item[1].get("shape_match", False))),
            int(bool(item[1].get("orientation_match", False))),
            int(item[1].get("nms_detections", 0)),
            int(item[1].get("threshold_candidates", 0)),
            float(item[1].get("max_score", 0.0)),
        ),
    )
    _update_decode_debug(
        debug,
        output_shape=original_shape,
        squeezed_shape=tuple(int(item) for item in array.shape),
        candidates_shape=tuple(int(item) for item in candidates_first.shape),
        expected_classes=expected_classes,
        prediction_columns=columns,
        candidate_layouts=layout_modes,
        selected_layout=best_stats.get("layout", ""),
        raw_candidates=int(best_stats.get("raw_candidates", 0)),
        max_score=float(best_stats.get("max_score", 0.0)),
        threshold_candidates=int(best_stats.get("threshold_candidates", 0)),
        nms_detections=int(best_stats.get("nms_detections", 0)),
        variants=[stats for _, stats in variants],
    )
    return best_detections


def _decode_xyxy_score_cls(
    array: Any,
    *,
    confidence_threshold: float,
    nms_threshold: float,
) -> tuple[list[InferenceDetection], dict[str, Any]]:
    import numpy as np

    scores = np.asarray(array[:, 4], dtype=np.float32)
    stats: dict[str, Any] = {
        "layout": "xyxy-score-cls",
        "raw_candidates": int(array.shape[0]),
        "max_score": float(np.max(scores)) if scores.size else 0.0,
        "threshold_candidates": 0,
        "nms_detections": 0,
        "class_id_like": _looks_like_class_ids(array[:, 5]) if array.shape[1] >= 6 else False,
    }
    if array.shape[1] < 6 or scores.size == 0:
        return [], stats
    selected_indices = _pre_nms_indices(scores, confidence_threshold, np=np)
    candidates: list[InferenceDetection] = []
    threshold_candidates = int(np.count_nonzero(scores >= confidence_threshold))
    for index in selected_indices:
        row = array[int(index)]
        score = float(row[4])
        x1, y1, x2, y2 = [float(value) for value in row[:4]]
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(
            InferenceDetection.from_xyxy(
                cls_id=int(round(float(row[5]))),
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    kept = _nms(candidates, nms_threshold)
    stats["threshold_candidates"] = int(threshold_candidates)
    stats["nms_input_candidates"] = int(len(candidates))
    stats["nms_detections"] = int(len(kept))
    return kept, stats


def _decode_yolo_scores(
    array: Any,
    *,
    mode: str,
    confidence_threshold: float,
    nms_threshold: float,
) -> tuple[list[InferenceDetection], dict[str, Any]]:
    import numpy as np

    boxes = array[:, :4]
    if mode == "yolov5-cxcywh-obj-cls":
        if array.shape[1] < 6:
            scores = np.empty((array.shape[0], 0), dtype=np.float32)
        else:
            obj = array[:, 4]
            scores = obj[:, None] * array[:, 5:]
    else:
        scores = array[:, 4:]
    stats: dict[str, Any] = {
        "layout": mode,
        "raw_candidates": int(array.shape[0]),
        "max_score": 0.0,
        "threshold_candidates": 0,
        "nms_detections": 0,
    }
    if scores.size == 0 or scores.shape[1] == 0:
        return [], stats
    cls_ids = np.argmax(scores, axis=1)
    cls_scores = np.max(scores, axis=1)
    stats["max_score"] = float(np.max(cls_scores)) if cls_scores.size else 0.0
    selected_indices = _pre_nms_indices(cls_scores, confidence_threshold, np=np)
    stats["threshold_candidates"] = int(np.count_nonzero(cls_scores >= confidence_threshold))
    candidates: list[InferenceDetection] = []
    for index in selected_indices:
        score = float(cls_scores[int(index)])
        cx, cy, w, h = [float(value) for value in boxes[int(index)]]
        if w <= 0 or h <= 0:
            continue
        candidates.append(
            InferenceDetection.from_cxcywh(
                cls_id=int(cls_ids[int(index)]),
                score=score,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
            )
        )
    kept = _nms(candidates, nms_threshold)
    stats["nms_input_candidates"] = int(len(candidates))
    stats["nms_detections"] = int(len(kept))
    return kept, stats


def _select_decode_layouts(
    *,
    columns: int,
    class_count: int,
    transposed_from_channel_first: bool,
) -> list[str]:
    if class_count > 0:
        if columns == 4 + class_count:
            return ["yolov8-cxcywh-cls"]
        if columns == 5 + class_count:
            return ["yolov5-cxcywh-obj-cls"]
    if transposed_from_channel_first:
        return ["yolov8-cxcywh-cls"]
    if columns >= 6:
        return ["yolov5-cxcywh-obj-cls", "xyxy-score-cls"]
    return ["yolov8-cxcywh-cls"]


def _pre_nms_indices(scores: Any, confidence_threshold: float, *, np: Any) -> Any:
    scores_array = np.asarray(scores, dtype=np.float32)
    indices = np.flatnonzero(scores_array >= confidence_threshold)
    if indices.size <= MAX_NMS_CANDIDATES:
        return indices
    ranked_local = np.argpartition(scores_array[indices], -MAX_NMS_CANDIDATES)[-MAX_NMS_CANDIDATES:]
    top_indices = indices[ranked_local]
    return top_indices[np.argsort(scores_array[top_indices])[::-1]]


def _update_decode_debug(debug: dict[str, Any] | None, **values: Any) -> None:
    if debug is None:
        return
    for key, value in values.items():
        if isinstance(value, tuple):
            debug[key] = [int(item) for item in value]
        else:
            debug[key] = value


def _looks_like_class_ids(values: Any) -> bool:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return False
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return False
    rounded = np.round(finite)
    return bool(np.all(np.abs(finite - rounded) <= 1e-3) and np.min(finite) >= 0)


def _layout_shape_matches(layout: str, *, columns: int, class_count: int) -> bool:
    if class_count <= 0:
        return False
    if layout == "yolov8-cxcywh-cls":
        return columns == 4 + class_count
    if layout == "yolov5-cxcywh-obj-cls":
        return columns == 5 + class_count
    return False


def _layout_orientation_matches(layout: str, *, transposed_from_channel_first: bool) -> bool:
    if layout == "yolov8-cxcywh-cls":
        return transposed_from_channel_first
    if layout == "yolov5-cxcywh-obj-cls":
        return not transposed_from_channel_first
    return False


def _nms(detections: list[InferenceDetection], threshold: float) -> list[InferenceDetection]:
    if not detections:
        return []
    import numpy as np

    kept_indices: list[int] = []
    classes = np.asarray([item.cls for item in detections], dtype=np.int32)
    scores = np.asarray([item.score for item in detections], dtype=np.float32)
    x1 = np.asarray([item.x for item in detections], dtype=np.float32)
    y1 = np.asarray([item.y for item in detections], dtype=np.float32)
    x2 = x1 + np.asarray([item.w for item in detections], dtype=np.float32)
    y2 = y1 + np.asarray([item.h for item in detections], dtype=np.float32)
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    for cls in np.unique(classes):
        remaining = np.where(classes == cls)[0]
        remaining = remaining[np.argsort(scores[remaining])[::-1]]
        while remaining.size:
            current = int(remaining[0])
            kept_indices.append(current)
            if remaining.size == 1:
                break
            rest = remaining[1:]
            xx1 = np.maximum(x1[current], x1[rest])
            yy1 = np.maximum(y1[current], y1[rest])
            xx2 = np.minimum(x2[current], x2[rest])
            yy2 = np.minimum(y2[current], y2[rest])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h
            union = areas[current] + areas[rest] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            remaining = rest[iou <= threshold]
    kept_indices.sort(key=lambda index: detections[index].score, reverse=True)
    return [detections[index] for index in kept_indices]
