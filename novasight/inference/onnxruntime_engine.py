from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

from novasight.coordinates import CoordinateTransform

from .contracts import InferenceDetection, InferenceResult
from .input import PreparedTensorInput, TensorInputShape, parse_tensor_input_shape, prepare_tensor_input
from .postprocess.yolo import decode_nx6_detections
from .preprocess import TensorPreprocessResult, prepare_tensor


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
        self._input_shape_source = ""
        self._session: Any | None = None
        self._input_name = ""
        self._last_input: PreparedTensorInput | None = None
        self._last_preprocess_backend = ""
        self._last_preprocess_reason = ""

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
            "last_input_frame_id": (
                self._last_input.frame_id if self._last_input is not None else 0
            ),
            "last_input_capture_ts_ns": (
                self._last_input.capture_ts_ns if self._last_input is not None else 0
            ),
            "last_input_resource_kind": (
                self._last_input.resource_kind if self._last_input is not None else ""
            ),
            "last_input_resource_memory": (
                self._last_input.resource_memory if self._last_input is not None else ""
            ),
            "last_input_resource_source": (
                self._last_input.resource_source if self._last_input is not None else ""
            ),
            "last_input_resource_size": (
                f"{self._last_input.resource_width}x{self._last_input.resource_height}"
                if self._last_input is not None
                and self._last_input.resource_width > 0
                and self._last_input.resource_height > 0
                else ""
            ),
            "last_input_resource_format": (
                self._last_input.resource_pixel_format if self._last_input is not None else ""
            ),
            "last_input_needs_resize": (
                self._last_input.needs_resize if self._last_input is not None else False
            ),
            "last_preprocess_backend": self._last_preprocess_backend,
            "last_preprocess_reason": self._last_preprocess_reason,
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
        self._last_preprocess_backend = ""
        self._last_preprocess_reason = ""
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
            preprocess_result = prepare_tensor(self._last_input, self._input_shape)
            self._last_preprocess_backend = preprocess_result.backend
            self._last_preprocess_reason = preprocess_result.reason
            tensor = preprocess_result.tensor
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
            detections = map_model_detections_to_roi_frame(
                detections,
                prepared=self._last_input,
                shape=self._input_shape,
            )
            preprocess_debug = _preprocess_debug(
                self._last_input,
                self._input_shape,
                preprocess_result=preprocess_result,
            )
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
            self._last_preprocess_backend = "unavailable"
            self._last_preprocess_reason = getattr(exc, "reason", "")
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


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return max(0.0, (end_ns - start_ns) / 1e6)


def _coordinate_transform_for_prepared_input(
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
    transform = _coordinate_transform_for_prepared_input(prepared, shape)
    return [
        InferenceDetection(
            cls=item.cls,
            score=item.score,
            box=transform.model_to_roi_box(item.box),
        )
        for item in detections
    ]


def _scale_detections_to_input_frame(
    detections: list[InferenceDetection],
    *,
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
) -> list[InferenceDetection]:
    return map_model_detections_to_roi_frame(
        detections,
        prepared=prepared,
        shape=shape,
    )


def _preprocess_debug(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    preprocess_result: TensorPreprocessResult | None = None,
) -> dict[str, Any]:
    transform = _coordinate_transform_for_prepared_input(prepared, shape)
    roi_width = max(1, prepared.width)
    roi_height = max(1, prepared.height)
    roi_pixels = max(1, roi_width * roi_height)
    model_pixels = max(1, shape.width * shape.height)
    model_to_roi_scale_x = transform.model_to_roi_scale_x
    model_to_roi_scale_y = transform.model_to_roi_scale_y
    downscale_factor = max(model_to_roi_scale_x, model_to_roi_scale_y)
    pixel_ratio = model_pixels / roi_pixels
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
        "pixel_ratio": pixel_ratio,
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
