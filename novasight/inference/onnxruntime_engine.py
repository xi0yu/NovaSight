from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

from .contracts import InferenceDetection, InferenceResult
from .input import PreparedTensorInput, TensorInputShape, parse_tensor_input_shape, prepare_tensor_input


SessionFactory = Callable[[Path], Any]
MAX_NMS_CANDIDATES = 300


class OnnxRuntimeInferenceEngine:
    engine_id = "onnxruntime"

    def __init__(
        self,
        *,
        session_factory: SessionFactory | None = None,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        self._session_factory = session_factory
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._reason = ""
        self._available = False
        self._loaded = False
        self._classes: list[str] = []
        self._input_shape: TensorInputShape | None = None
        self._input_shape_source = ""
        self._session: Any | None = None
        self._input_name = ""
        self._last_input: PreparedTensorInput | None = None

    def available(self) -> bool:
        if self._session_factory is not None:
            self._available = True
            self._reason = ""
            return True
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            self._available = False
            self._reason = f"ONNX Runtime unavailable: {exc}"
            return False
        self._available = True
        self._reason = ""
        return True

    def last_reason(self) -> str:
        return self._reason

    def status(self) -> dict:
        return {
            "selected": self.engine_id,
            "available": self._available,
            "loaded": self._loaded,
            "reason": self._reason,
            "input_shape": str(self._input_shape) if self._input_shape is not None else "",
            "input_shape_source": self._input_shape_source,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "last_input_mode": self._last_input.mode if self._last_input is not None else "",
            "last_input_needs_resize": (
                self._last_input.needs_resize if self._last_input is not None else False
            ),
        }

    def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
        if artifact_path.suffix.lower() != ".onnx":
            raise ValueError(f"ONNX artifact must be .onnx: {artifact_path}")
        if not classes:
            raise ValueError("ONNX classes must not be empty")
        configured_shape = parse_tensor_input_shape(input_shape) if input_shape.strip() else None
        if not self.available():
            raise RuntimeError(self._reason)
        self._available = True
        self._session = self._create_session(artifact_path)
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no inputs")
        model_input = inputs[0]
        self._input_name = str(model_input.name)
        self._classes = list(classes)
        self._input_shape, self._input_shape_source = _resolve_onnx_input_shape(
            model_input.shape,
            configured_shape=configured_shape,
        )
        self._last_input = None
        self._loaded = True

    def infer(self, frame: Any) -> InferenceResult:
        if not self._loaded or self._session is None:
            return InferenceResult(available=False, reason="ONNX model not loaded")
        if self._input_shape is None:
            return InferenceResult(available=False, reason="ONNX input shape not loaded")
        timings: dict[str, float] = {}
        total_start_ns = time.monotonic_ns()
        try:
            prepare_start_ns = time.monotonic_ns()
            self._last_input = prepare_tensor_input(frame, self._input_shape)
            tensor_start_ns = time.monotonic_ns()
            tensor = _prepare_numpy_tensor(self._last_input, self._input_shape)
            run_start_ns = time.monotonic_ns()
            outputs = self._session.run(None, {self._input_name: tensor})
            decode_start_ns = time.monotonic_ns()
            primary_output = outputs[0] if outputs else []
            decode_debug: dict[str, Any] = {}
            detections = decode_nx6_detections(
                primary_output,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                class_count=len(self._classes),
                debug=decode_debug,
            )
            scale_start_ns = time.monotonic_ns()
            detections = _scale_detections_to_input_frame(
                detections,
                prepared=self._last_input,
                shape=self._input_shape,
            )
            preprocess_debug = _preprocess_debug(self._last_input, self._input_shape)
            done_ns = time.monotonic_ns()
            timings.update(
                {
                    "prepare_input_ms": _elapsed_ms(prepare_start_ns, tensor_start_ns),
                    "numpy_tensor_ms": _elapsed_ms(tensor_start_ns, run_start_ns),
                    "execute_total_ms": _elapsed_ms(run_start_ns, decode_start_ns),
                    "decode_ms": _elapsed_ms(decode_start_ns, scale_start_ns),
                    "scale_ms": _elapsed_ms(scale_start_ns, done_ns),
                    "total_ms": _elapsed_ms(total_start_ns, done_ns),
                }
            )
        except Exception as exc:
            return InferenceResult(available=False, reason=str(exc))
        output_shape = tuple(getattr(primary_output, "shape", ()))
        return InferenceResult(
            available=True,
            detections=detections,
            classes=self._classes,
            debug={
                "engine": self.engine_id,
                "input_name": self._input_name,
                "output_shape": list(output_shape),
                "decoded_detections": len(detections),
                "preprocess": preprocess_debug,
                "decode": decode_debug,
                "timings": timings,
            },
        )

    def _create_session(self, artifact_path: Path) -> Any:
        if self._session_factory is not None:
            return self._session_factory(artifact_path)
        import onnxruntime

        available = set(onnxruntime.get_available_providers())
        preferred = [
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [provider for provider in preferred if provider in available]
        if not providers:
            raise RuntimeError("ONNX Runtime has no available execution providers")
        return onnxruntime.InferenceSession(
            str(artifact_path),
            providers=providers,
        )


def _prepare_numpy_tensor(prepared: PreparedTensorInput, shape: TensorInputShape) -> Any:
    import numpy as np

    if prepared.mode == "gpu_buffer":
        return np.zeros((shape.batch, shape.channels, shape.height, shape.width), dtype=np.float32)
    image = prepared.buffer
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            if image.size != (shape.width, shape.height):
                image = image.resize((shape.width, shape.height))
            array = np.asarray(image, dtype=np.float32) / 255.0
            return array.transpose(2, 0, 1)[None, ...]
    except Exception:
        pass
    if hasattr(image, "shape"):
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim == 3 and array.shape[2] >= 3:
            array = np.ascontiguousarray(array[:, :, :3])
            if prepared.pixel_format in {"BGR", "BGR3"}:
                array = array[:, :, ::-1]
            if array.shape[1] != shape.width or array.shape[0] != shape.height:
                array = _resize_numpy_image(array, width=shape.width, height=shape.height, np=np)
            array = array.astype(np.float32) / 255.0
            return array.transpose(2, 0, 1)[None, ...]
    return np.zeros((shape.batch, shape.channels, shape.height, shape.width), dtype=np.float32)


def _resolve_onnx_input_shape(
    model_shape: Any,
    *,
    configured_shape: TensorInputShape | None,
) -> tuple[TensorInputShape, str]:
    static_shape = _parse_static_onnx_shape(model_shape)
    if static_shape is not None:
        return static_shape, "model_static"
    if configured_shape is not None:
        return configured_shape, "configured_dynamic_model"
    raise ValueError(f"ONNX model input shape is dynamic and no runtime shape was provided: {model_shape}")


def _parse_static_onnx_shape(model_shape: Any) -> TensorInputShape | None:
    try:
        values = [int(item) for item in model_shape]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or min(values) <= 0:
        return None
    batch, channels, height, width = values
    if channels not in {1, 3, 4}:
        return None
    return TensorInputShape(
        batch=batch,
        channels=channels,
        height=height,
        width=width,
    )


def _resize_numpy_image(image: Any, *, width: int, height: int, np: Any) -> Any:
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((width, height)))
    except Exception:
        y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
        x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
        return image[y_idx][:, x_idx]


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return max(0.0, (end_ns - start_ns) / 1e6)


def _scale_detections_to_input_frame(
    detections: list[InferenceDetection],
    *,
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
) -> list[InferenceDetection]:
    if not detections:
        return []
    scale_x = prepared.width / shape.width
    scale_y = prepared.height / shape.height
    if scale_x == 1 and scale_y == 1:
        return detections
    return [
        InferenceDetection(
            cls=item.cls,
            score=item.score,
            x=item.x * scale_x,
            y=item.y * scale_y,
            w=item.w * scale_x,
            h=item.h * scale_y,
        )
        for item in detections
    ]


def _preprocess_debug(prepared: PreparedTensorInput, shape: TensorInputShape) -> dict[str, Any]:
    roi_width = max(1, prepared.width)
    roi_height = max(1, prepared.height)
    roi_pixels = max(1, roi_width * roi_height)
    model_pixels = max(1, shape.width * shape.height)
    model_to_roi_scale_x = roi_width / shape.width
    model_to_roi_scale_y = roi_height / shape.height
    downscale_factor = max(model_to_roi_scale_x, model_to_roi_scale_y)
    pixel_ratio = model_pixels / roi_pixels
    return {
        "input_mode": prepared.mode,
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
        "pixel_ratio": pixel_ratio,
        "density_warning": prepared.needs_resize and downscale_factor > 1.25,
    }


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
            InferenceDetection(
                cls=int(round(float(row[5]))),
                score=score,
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
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
        score_value = cls_scores[int(index)]
        score = float(score_value)
        cx, cy, w, h = [float(value) for value in boxes[int(index)]]
        if w <= 0 or h <= 0:
            continue
        candidates.append(
            InferenceDetection(
                cls=int(cls_ids[int(index)]),
                score=score,
                x=cx - w / 2,
                y=cy - h / 2,
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


def _iou(left: InferenceDetection, right: InferenceDetection) -> float:
    left_x2 = left.x + left.w
    left_y2 = left.y + left.h
    right_x2 = right.x + right.w
    right_y2 = right.y + right.h
    ix1 = max(left.x, right.x)
    iy1 = max(left.y, right.y)
    ix2 = min(left_x2, right_x2)
    iy2 = min(left_y2, right_y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = left.w * left.h + right.w * right.h - intersection
    if union <= 0:
        return 0.0
    return intersection / union
