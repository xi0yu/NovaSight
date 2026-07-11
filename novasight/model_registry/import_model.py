from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any

from .manifest import (
    ModelManifest,
    TensorSpec,
    build_engine_manifest,
    write_manifest,
)


@dataclass(frozen=True)
class ImportedModel:
    manifest: ModelManifest
    model_path: Path
    manifest_path: Path


def import_onnx_model(
    source_path: Path,
    *,
    output_dir: Path = Path("models/originals"),
    target_dir: Path | None = None,
    model_id: str | None = None,
    display_name: str | None = None,
    class_names: list[str] | None = None,
    confidence_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    runtime_precision: str = "fp16",
) -> ImportedModel:
    source_path = Path(source_path)
    if source_path.suffix.lower() != ".onnx":
        raise ValueError(f"import_onnx_model requires an .onnx file: {source_path}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    inferred = inspect_onnx_model(source_path)
    model_name = _safe_component(model_id or source_path.stem)
    resolved_target_dir = Path(target_dir) if target_dir is not None else Path(output_dir) / model_name
    resolved_target_dir.mkdir(parents=True, exist_ok=True)
    target_path = resolved_target_dir / source_path.name
    if source_path.resolve(strict=False) != target_path.resolve(strict=False):
        shutil.copy2(source_path, target_path)

    classes = list(class_names or inferred.class_names)
    class_count = len(classes) if classes else inferred.class_count
    if inferred.postprocess_parser != "yolo" and not classes:
        raise ValueError(
            "model output appears to contain Decode/NMS; explicit class names are required"
        )
    if class_count <= 0:
        class_count = max(1, infer_yolo_class_count(inferred.output.shape))
    if not classes:
        classes = [f"class_{index}" for index in range(class_count)]

    manifest = build_engine_manifest(
        model_id=model_name,
        display_name=display_name or source_path.stem,
        engine_path=target_path,
        input_spec=inferred.input,
        output_spec=inferred.output,
        class_count=class_count,
        class_names=classes,
        confidence_threshold=float(confidence_threshold),
        nms_iou_threshold=float(nms_iou_threshold),
        runtime_precision=runtime_precision,
        output_format=inferred.output_format,
        output_has_objectness=inferred.output_has_objectness,
        postprocess_parser=inferred.postprocess_parser,
        validated=False,
    )
    manifest_path = resolved_target_dir / "model.manifest.json"
    write_manifest(manifest, manifest_path)
    return ImportedModel(manifest=manifest, model_path=target_path, manifest_path=manifest_path)


@dataclass(frozen=True)
class OnnxModelInspection:
    input: TensorSpec
    output: TensorSpec
    class_count: int
    class_names: list[str]
    output_format: str = "yolo_cxcywh_class_scores"
    output_has_objectness: bool = False
    postprocess_parser: str = "yolo"


def inspect_onnx_model(path: Path) -> OnnxModelInspection:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is required to inspect ONNX model files") from exc

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs:
        raise ValueError(f"ONNX model has no graph inputs: {path}")
    if not outputs:
        raise ValueError(f"ONNX model has no graph outputs: {path}")
    input_meta = inputs[0]
    if len(outputs) != 1:
        names = ", ".join(str(item.name) for item in outputs)
        raise ValueError(
            "multi-output ONNX models require an explicit output contract; "
            f"refusing to guess Decode/NMS bindings: {names}"
        )
    output_meta = outputs[0]
    input_shape = _normalize_shape(input_meta.shape, fallback=[1, 3, 640, 640])
    output_shape = _normalize_shape(output_meta.shape, fallback=[1, 84, 8400])
    built_in_nms = _looks_like_single_tensor_nms(output_shape)
    class_count = 0 if built_in_nms else infer_yolo_class_count(output_shape)
    return OnnxModelInspection(
        input=TensorSpec(
            name=str(input_meta.name),
            shape=input_shape,
            dtype=_normalize_onnxruntime_dtype(input_meta.type),
            layout="NCHW",
        ),
        output=TensorSpec(
            name=str(output_meta.name),
            shape=output_shape,
            dtype=_normalize_onnxruntime_dtype(output_meta.type),
            layout="NCHW",
        ),
        class_count=class_count,
        class_names=[],
        output_format=("xyxy_score_class" if built_in_nms else "yolo_cxcywh_class_scores"),
        output_has_objectness=False,
        postprocess_parser=("efficientnms" if built_in_nms else "yolo"),
    )


def _looks_like_single_tensor_nms(shape: list[int]) -> bool:
    if len(shape) == 3 and shape[0] == 1:
        return shape[-1] == 6 and 1 <= shape[-2] <= 512
    if len(shape) == 2:
        return shape[-1] == 6 and 1 <= shape[-2] <= 512
    return False


def infer_yolo_class_count(shape: list[int]) -> int:
    if len(shape) != 3:
        return 0
    feature_dim = min(shape[1], shape[2])
    return feature_dim - 4 if feature_dim > 4 else 0


def _normalize_shape(value: Any, *, fallback: list[int]) -> list[int]:
    if not isinstance(value, list):
        return list(fallback)
    normalized: list[int] = []
    for index, item in enumerate(value):
        try:
            dim = int(item)
        except (TypeError, ValueError):
            dim = int(fallback[index]) if index < len(fallback) else 1
        normalized.append(dim if dim > 0 else int(fallback[index]) if index < len(fallback) else 1)
    return normalized or list(fallback)


def _normalize_onnxruntime_dtype(value: Any) -> str:
    text = str(value or "").lower()
    if "float16" in text:
        return "float16"
    if "float" in text:
        return "float32"
    if "int64" in text:
        return "int64"
    if "int32" in text:
        return "int32"
    if "uint8" in text:
        return "uint8"
    return text.removeprefix("tensor(").removesuffix(")") or "float32"


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not normalized:
        raise ValueError("model_id must contain at least one safe path character")
    return normalized


__all__ = [
    "ImportedModel",
    "OnnxModelInspection",
    "import_onnx_model",
    "infer_yolo_class_count",
    "inspect_onnx_model",
]
