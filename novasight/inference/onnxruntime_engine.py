from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .contracts import InferenceDetection, InferenceResult
from .input import PreparedTensorInput, TensorInputShape, parse_tensor_input_shape, prepare_tensor_input


SessionFactory = Callable[[Path], Any]


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
        parsed_shape = parse_tensor_input_shape(input_shape)
        if not self.available():
            raise RuntimeError(self._reason)
        self._available = True
        self._session = self._create_session(artifact_path)
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no inputs")
        self._input_name = str(inputs[0].name)
        self._classes = list(classes)
        self._input_shape = parsed_shape
        self._last_input = None
        self._loaded = True

    def infer(self, frame: Any) -> InferenceResult:
        if not self._loaded or self._session is None:
            return InferenceResult(available=False, reason="ONNX model not loaded")
        if self._input_shape is None:
            return InferenceResult(available=False, reason="ONNX input shape not loaded")
        try:
            self._last_input = prepare_tensor_input(frame, self._input_shape)
            tensor = _prepare_numpy_tensor(self._last_input, self._input_shape)
            outputs = self._session.run(None, {self._input_name: tensor})
            primary_output = outputs[0] if outputs else []
            decode_debug: dict[str, Any] = {}
            detections = decode_nx6_detections(
                primary_output,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                debug=decode_debug,
            )
            detections = _scale_detections_to_input_frame(
                detections,
                prepared=self._last_input,
                shape=self._input_shape,
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
                "decode": decode_debug,
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


def _resize_numpy_image(image: Any, *, width: int, height: int, np: Any) -> Any:
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((width, height)))
    except Exception:
        y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
        x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
        return image[y_idx][:, x_idx]


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


def decode_nx6_detections(
    output: Any,
    *,
    confidence_threshold: float,
    nms_threshold: float,
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

    candidates_first = (
        array.T
        if 4 < array.shape[0] < array.shape[1]
        else array
    )
    if candidates_first.shape[1] < 5:
        _update_decode_debug(
            debug,
            output_shape=original_shape,
            squeezed_shape=tuple(int(item) for item in array.shape),
            candidates_shape=tuple(int(item) for item in candidates_first.shape),
            reason="fewer than 5 prediction columns",
        )
        return []

    variants: list[tuple[list[InferenceDetection], dict[str, Any]]] = []
    variants.append(
        _decode_yolo_scores(
            candidates_first,
            mode="yolov8-cxcywh-cls",
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )
    )
    if candidates_first.shape[1] >= 6:
        variants.append(
            _decode_yolo_scores(
                candidates_first,
                mode="yolov5-cxcywh-obj-cls",
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
            )
        )
        variants.append(
            _decode_xyxy_score_cls(
                candidates_first,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
            )
        )
    best_detections, best_stats = max(
        variants,
        key=lambda item: (
            int(
                bool(item[1].get("class_id_like", False))
                and int(item[1].get("nms_detections", 0)) > 0
            ),
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
    candidates: list[InferenceDetection] = []
    threshold_candidates = 0
    for row in array:
        score = float(row[4])
        if score < confidence_threshold:
            continue
        threshold_candidates += 1
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
    threshold_mask = cls_scores >= confidence_threshold
    stats["threshold_candidates"] = int(np.count_nonzero(threshold_mask))
    candidates: list[InferenceDetection] = []
    for index, score_value in enumerate(cls_scores):
        score = float(score_value)
        if score < confidence_threshold:
            continue
        cx, cy, w, h = [float(value) for value in boxes[index]]
        if w <= 0 or h <= 0:
            continue
        candidates.append(
            InferenceDetection(
                cls=int(cls_ids[index]),
                score=score,
                x=cx - w / 2,
                y=cy - h / 2,
                w=w,
                h=h,
            )
        )
    kept = _nms(candidates, nms_threshold)
    stats["nms_detections"] = int(len(kept))
    return kept, stats


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


def _nms(detections: list[InferenceDetection], threshold: float) -> list[InferenceDetection]:
    kept: list[InferenceDetection] = []
    for item in sorted(detections, key=lambda detection: detection.score, reverse=True):
        if all(item.cls != kept_item.cls or _iou(item, kept_item) <= threshold for kept_item in kept):
            kept.append(item)
    return kept


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
