from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novasight.inference.input import normalize_tensor_dtype
from novasight.model_registry.manifest import (
    ModelManifest,
    TensorSpec,
    build_engine_manifest,
    read_manifest,
    validate_manifest_engine_artifact,
    write_manifest,
)


_MANIFEST_LOCK = threading.RLock()
logger = logging.getLogger("novasight.deepstream.model_manifest")


@dataclass(frozen=True, slots=True)
class EngineTensorContract:
    input_name: str
    input_shape: list[int]
    input_dtype: str
    output_name: str
    output_shape: list[int]
    output_dtype: str


def probe_engine_contract(
    inference: Any,
    *,
    artifact_path: Path,
    classes: list[str],
    registered_input_shape: str,
) -> EngineTensorContract:
    probe = getattr(inference, "probe", None)
    if not callable(probe):
        raise ValueError(
            "TensorRT engine probe is unavailable; refusing to guess DeepStream tensor bindings"
        )
    status = dict(probe(artifact_path, classes, registered_input_shape))
    if status.get("loaded") is not True:
        raise ValueError(
            f"TensorRT engine probe failed: {status.get('reason') or 'loaded=false'}"
        )
    outputs = status.get("outputs")
    if isinstance(outputs, dict) and len(outputs) != 1:
        raise ValueError(
            "DeepStream native parser requires exactly one TensorRT output tensor, "
            f"got {sorted(str(name) for name in outputs)}"
        )
    input_name = str(status.get("input_name") or "").strip()
    output_name = str(status.get("output_name") or "").strip()
    if not input_name or not output_name:
        raise ValueError("TensorRT engine probe did not expose input/output tensor names")
    return EngineTensorContract(
        input_name=input_name,
        input_shape=parse_runtime_shape(status.get("input_shape"), "input_shape"),
        input_dtype=normalize_tensor_dtype(status.get("input_dtype")),
        output_name=output_name,
        output_shape=parse_runtime_shape(status.get("output_shape"), "output_shape"),
        output_dtype=normalize_tensor_dtype(status.get("output_dtype")),
    )


def infer_yolo_output_contract(output_shape: list[int], class_count: int) -> bool:
    classes = int(class_count)
    if classes <= 0:
        raise ValueError(f"DeepStream class_count must be positive, got {classes}")
    shape, channels, _candidates = _raw_yolo_dimensions(output_shape)
    if channels not in {4 + classes, 5 + classes}:
        raise ValueError(
            "DeepStream YOLO output must have a channel dimension equal to "
            "4 + class_count or 5 + class_count "
            f"(shape={shape}, class_count={classes})"
        )
    return channels == 5 + classes


def resolve_yolo_class_contract(
    output_shape: list[int],
    registered_classes: list[str],
    *,
    class_count_hint: int | None = None,
) -> tuple[list[str], bool]:
    classes = [str(item).strip() for item in registered_classes if str(item).strip()]
    automatic_classes = _automatic_class_names(classes)
    if class_count_hint is not None and automatic_classes:
        count = int(class_count_hint)
        has_objectness = infer_yolo_output_contract(output_shape, count)
        return [f"class_{index}" for index in range(count)], has_objectness
    if not automatic_classes:
        return classes, infer_yolo_output_contract(output_shape, len(classes))
    try:
        return classes, infer_yolo_output_contract(output_shape, len(classes))
    except ValueError:
        pass
    _shape, channels, _candidates = _raw_yolo_dimensions(output_shape)
    inferred_class_count = channels - 4
    if inferred_class_count <= 0:
        raise ValueError(
            "cannot infer Raw YOLO class count from placeholder model metadata "
            f"(shape={output_shape})"
        )
    return [f"class_{index}" for index in range(inferred_class_count)], False


def infer_class_count_hint_from_name(value: str) -> int | None:
    text = Path(str(value)).stem
    numeric = re.search(r"(?<!\d)(\d{1,3})\s*(?:类|classes?|class|cls)", text, re.IGNORECASE)
    if numeric is not None:
        count = int(numeric.group(1))
        return count if count > 0 else None
    chinese = re.search(r"([零〇一二两三四五六七八九十]+)\s*类", text)
    if chinese is None:
        return None
    return _parse_chinese_integer(chinese.group(1))


def _parse_chinese_integer(value: str) -> int | None:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if "十" not in value:
        count = digits.get(value)
        return count if count and count > 0 else None
    left, right = value.split("十", 1)
    tens = digits.get(left, 1) if left else 1
    ones = digits.get(right, 0) if right else 0
    count = tens * 10 + ones
    return count if count > 0 else None


def _automatic_class_names(classes: list[str]) -> bool:
    if classes == ["target"]:
        return True
    return bool(classes) and all(name == f"class_{index}" for index, name in enumerate(classes))


def _manifest_needs_class_hint_reconciliation(
    manifest: ModelManifest,
    class_count_hint: int | None,
) -> bool:
    if class_count_hint is None or not _automatic_class_names(list(manifest.output.class_names)):
        return False
    expected_objectness = infer_yolo_output_contract(
        list(manifest.output.shape),
        int(class_count_hint),
    )
    return (
        int(manifest.output.class_count) != int(class_count_hint)
        or bool(manifest.output.has_objectness) != expected_objectness
    )


def _raw_yolo_dimensions(output_shape: list[int]) -> tuple[list[int], int, int]:
    shape = [int(item) for item in output_shape]
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(
            "DeepStream YOLO output must be [1, channels, candidates] "
            f"or [1, candidates, channels], got {shape}"
        )
    first, second = shape[1], shape[2]
    if 6 in {first, second} and (second if first == 6 else first) <= 512:
        raise ValueError(
            "TensorRT output looks like built-in Decode/NMS [1,N,6]; "
            "the raw YOLO parser contract cannot be generated automatically"
        )
    channels = min(first, second)
    candidates = max(first, second)
    if channels <= 4 or candidates <= channels:
        raise ValueError(
            "DeepStream YOLO output candidate dimension must exceed a channel "
            f"dimension greater than four (shape={shape})"
        )
    return shape, channels, candidates


def ensure_engine_manifest(
    inference: Any,
    *,
    engine_path: Path,
    model_id: str,
    display_name: str,
    classes: list[str],
    registered_input_shape: str,
    confidence_threshold: float,
    nms_iou_threshold: float,
    runtime_precision: str = "fp16",
) -> tuple[ModelManifest, bool]:
    path = Path(engine_path)
    manifest_path = path.with_name("model.manifest.json")
    class_count_hint = infer_class_count_hint_from_name(path.name)
    with _MANIFEST_LOCK:
        if manifest_path.is_file():
            manifest = read_manifest(manifest_path)
            validate_manifest_engine_artifact(manifest, path)
            if not _manifest_needs_class_hint_reconciliation(manifest, class_count_hint):
                return manifest, False
            logger.warning(
                "regenerating automatic DeepStream manifest from engine filename class hint "
                "path=%s previous_classes=%s hinted_classes=%s",
                path,
                manifest.output.class_count,
                class_count_hint,
            )
        if not classes:
            raise ValueError(
                "cannot generate DeepStream manifest because the model registry has no classes"
            )
        contract = probe_engine_contract(
            inference,
            artifact_path=path,
            classes=classes,
            registered_input_shape=registered_input_shape,
        )
        if (
            len(contract.input_shape) != 4
            or contract.input_shape[0] != 1
            or contract.input_shape[1] != 3
        ):
            raise ValueError(
                "DeepStream model input must be static NCHW [1,3,H,W], "
                f"got {contract.input_shape}"
            )
        resolved_classes, output_has_objectness = resolve_yolo_class_contract(
            contract.output_shape,
            classes,
            class_count_hint=class_count_hint,
        )
        manifest = build_engine_manifest(
            model_id=model_id,
            display_name=display_name,
            engine_path=path,
            input_spec=TensorSpec(
                name=contract.input_name,
                shape=contract.input_shape,
                dtype=contract.input_dtype,
                layout="NCHW",
            ),
            output_spec=TensorSpec(
                name=contract.output_name,
                shape=contract.output_shape,
                dtype=contract.output_dtype,
                layout="NCHW",
            ),
            class_count=len(resolved_classes),
            class_names=resolved_classes,
            confidence_threshold=float(confidence_threshold),
            nms_iou_threshold=float(nms_iou_threshold),
            runtime_precision=runtime_precision,
            input_color_format="RGB",
            input_scale_factor=1.0 / 255.0,
            output_has_objectness=output_has_objectness,
            validated=True,
        )
        temporary_path = manifest_path.with_suffix(".json.tmp")
        try:
            write_manifest(manifest, temporary_path)
            temporary_path.replace(manifest_path)
            validate_manifest_engine_artifact(manifest, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise
        return manifest, True


def parse_runtime_shape(value: object, label: str) -> list[int]:
    parts = [item for item in re.split(r"[xX,\s]+", str(value or "").strip()) if item]
    try:
        shape = [int(item) for item in parts]
    except ValueError as exc:
        raise ValueError(f"TensorRT engine probe returned invalid {label}: {value}") from exc
    if not shape or any(item <= 0 for item in shape):
        raise ValueError(f"TensorRT engine probe returned unresolved {label}: {value}")
    return shape


__all__ = [
    "EngineTensorContract",
    "ensure_engine_manifest",
    "infer_yolo_output_contract",
    "infer_class_count_hint_from_name",
    "parse_runtime_shape",
    "probe_engine_contract",
    "resolve_yolo_class_contract",
]
