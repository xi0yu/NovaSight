from __future__ import annotations

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
    shape = [int(item) for item in output_shape]
    classes = int(class_count)
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(
            "DeepStream YOLO output must be [1, channels, candidates] "
            f"or [1, candidates, channels], got {shape}"
        )
    if classes <= 0:
        raise ValueError(f"DeepStream class_count must be positive, got {classes}")
    channel_dimensions = {4 + classes, 5 + classes}
    matches = [
        (index, value)
        for index, value in enumerate(shape[1:], start=1)
        if value in channel_dimensions
    ]
    if len(matches) != 1:
        raise ValueError(
            "DeepStream YOLO output must have exactly one dimension equal to "
            "4 + class_count or 5 + class_count "
            f"(shape={shape}, class_count={classes})"
        )
    channel_index, channels = matches[0]
    candidate_count = shape[2 if channel_index == 1 else 1]
    if candidate_count <= channels:
        raise ValueError(
            "DeepStream YOLO output candidate dimension must exceed channel dimension "
            f"(shape={shape})"
        )
    if shape[-1] == 6 and shape[-2] <= 512:
        raise ValueError(
            "TensorRT output looks like built-in Decode/NMS [1,N,6]; "
            "the raw YOLO parser contract cannot be generated automatically"
        )
    return channels == 5 + classes


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
    with _MANIFEST_LOCK:
        if manifest_path.is_file():
            manifest = read_manifest(manifest_path)
            validate_manifest_engine_artifact(manifest, path)
            return manifest, False
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
        output_has_objectness = infer_yolo_output_contract(
            contract.output_shape,
            len(classes),
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
            class_count=len(classes),
            class_names=classes,
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
    "parse_runtime_shape",
    "probe_engine_contract",
]
