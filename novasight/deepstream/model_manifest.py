from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from novasight.inference.input import normalize_tensor_dtype
from novasight.model_registry.manifest import (
    ModelManifest,
    TensorSpec,
    build_engine_manifest,
    read_manifest,
    validate_manifest_engine_artifact,
)

from .parser_presets import (
    ParserPlan,
    normalize_parser_preset,
    parser_preset_objectness_hint,
    resolve_parser_plan,
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


@dataclass(frozen=True, slots=True)
class EngineManifestRecommendation:
    contract: EngineTensorContract
    class_names: list[str]
    output_has_objectness: bool
    parser_plan: ParserPlan


def _write_manifest_preserving_extensions(
    manifest: ModelManifest,
    path: Path,
    *,
    source_path: Path | None,
) -> None:
    payload = manifest.to_dict()
    if source_path is not None and Path(source_path).is_file():
        try:
            source_payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            source_payload = None
        if isinstance(source_payload, dict):
            for key, value in source_payload.items():
                if key not in payload:
                    payload[key] = value
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    io_tensors = status.get("io_tensors")
    if isinstance(io_tensors, list):
        return _contract_from_io_tensors(io_tensors)
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


def _contract_from_io_tensors(io_tensors: list[object]) -> EngineTensorContract:
    inputs: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    for raw_tensor in io_tensors:
        if not isinstance(raw_tensor, dict):
            raise ValueError("TensorRT engine probe returned a malformed I/O tensor entry")
        mode = str(raw_tensor.get("mode") or "").strip().lower()
        if mode == "input":
            inputs.append(raw_tensor)
        elif mode == "output":
            outputs.append(raw_tensor)
        else:
            raise ValueError(
                "TensorRT engine probe returned an I/O tensor without input/output mode"
            )
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "DeepStream native parser requires exactly one TensorRT input and one output "
            f"tensor, got inputs={len(inputs)} outputs={len(outputs)}"
        )
    input_tensor = inputs[0]
    output_tensor = outputs[0]
    input_name = str(input_tensor.get("name") or "").strip()
    output_name = str(output_tensor.get("name") or "").strip()
    if not input_name or not output_name:
        raise ValueError("TensorRT engine probe returned an unnamed I/O tensor")
    return EngineTensorContract(
        input_name=input_name,
        input_shape=parse_runtime_shape(input_tensor.get("shape"), "input_shape"),
        input_dtype=normalize_tensor_dtype(input_tensor.get("dtype")),
        output_name=output_name,
        output_shape=parse_runtime_shape(output_tensor.get("shape"), "output_shape"),
        output_dtype=normalize_tensor_dtype(output_tensor.get("dtype")),
    )


def recommend_engine_manifest(
    inference: Any,
    *,
    artifact_path: Path,
    registered_classes: list[str],
    registered_input_shape: str,
    parser_preset: object = "auto",
) -> EngineManifestRecommendation:
    path = Path(artifact_path)
    contract = probe_engine_contract(
        inference,
        artifact_path=path,
        classes=registered_classes,
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
    preset_id = normalize_parser_preset(parser_preset)
    preset_objectness = parser_preset_objectness_hint(preset_id)
    resolved_classes, output_has_objectness = resolve_yolo_class_contract(
        contract.output_shape,
        registered_classes,
        class_count_hint=infer_class_count_hint_from_name(path.name),
        objectness_hint=(
            preset_objectness
            if preset_objectness is not None
            else infer_yolo_objectness_hint_from_name(path.name)
        ),
    )
    parser_plan = resolve_parser_plan(
        preset_id,
        output_shape=contract.output_shape,
        class_count=len(resolved_classes),
        inferred_has_objectness=output_has_objectness,
    )
    return EngineManifestRecommendation(
        contract=contract,
        class_names=resolved_classes,
        output_has_objectness=parser_plan.has_objectness,
        parser_plan=parser_plan,
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
    objectness_hint: bool | None = None,
) -> tuple[list[str], bool]:
    classes = [str(item).strip() for item in registered_classes if str(item).strip()]
    automatic_classes = _automatic_class_names(classes)
    if objectness_hint is not None and automatic_classes:
        _shape, channels, _candidates = _raw_yolo_dimensions(output_shape)
        count = channels - (5 if objectness_hint else 4)
        if count <= 0:
            raise ValueError(
                "cannot infer class count from the named YOLO output contract "
                f"(shape={output_shape}, has_objectness={objectness_hint})"
            )
        return [f"class_{index}" for index in range(count)], objectness_hint
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


def infer_yolo_objectness_hint_from_name(value: str) -> bool | None:
    text = Path(str(value)).stem.lower()
    if re.search(r"(?:yolo)?v(?:8|9|10|11|12)(?:[nslmx]|\d|[_-]|$)", text):
        return False
    if re.search(r"(?:yolo)?v5(?:[nslmx]|\d|[_-]|$)", text):
        return True
    return None


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
    objectness_hint: bool | None,
) -> bool:
    if not _automatic_class_names(list(manifest.output.class_names)):
        return False
    expected_classes, expected_objectness = resolve_yolo_class_contract(
        list(manifest.output.shape),
        list(manifest.output.class_names),
        class_count_hint=class_count_hint,
        objectness_hint=objectness_hint,
    )
    return (
        int(manifest.output.class_count) != len(expected_classes)
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
    parser_preset: object = "auto",
) -> tuple[ModelManifest, bool]:
    path = Path(engine_path)
    preset_id = normalize_parser_preset(parser_preset)
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    legacy_manifest_path = path.with_name("model.manifest.json")
    class_count_hint = infer_class_count_hint_from_name(path.name)
    objectness_hint = infer_yolo_objectness_hint_from_name(path.name)
    with _MANIFEST_LOCK:
        existing_manifest: ModelManifest | None = None
        existing_manifest_path: Path | None = None
        if manifest_path.is_file():
            existing_manifest_path = manifest_path
        elif legacy_manifest_path.is_file():
            existing_manifest_path = legacy_manifest_path
        recovered_profile_classes: list[str] = []
        if existing_manifest_path is not None:
            try:
                existing_manifest = read_manifest(existing_manifest_path)
                validate_manifest_engine_artifact(existing_manifest, path)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                recovered_profile_classes = _model_profile_class_names(
                    existing_manifest_path
                )
                existing_manifest = None
                logger.warning(
                    "replacing unusable model sidecar from TensorRT engine contract "
                    "path=%s sidecar=%s error=%s",
                    path,
                    existing_manifest_path,
                    exc,
                )

        probe_classes = [str(item) for item in classes if str(item).strip()]
        if (
            recovered_profile_classes
            and _automatic_class_names(probe_classes)
            and not _automatic_class_names(recovered_profile_classes)
        ):
            probe_classes = recovered_profile_classes
        if (
            existing_manifest is not None
            and _automatic_class_names(probe_classes)
            and not _automatic_class_names(list(existing_manifest.output.class_names))
        ):
            probe_classes = list(existing_manifest.output.class_names)
        if not probe_classes:
            probe_classes = ["target"]
        if existing_manifest is not None and not callable(getattr(inference, "probe", None)):
            registered_classes_match = (
                _automatic_class_names(probe_classes)
                or list(existing_manifest.output.class_names) == probe_classes
            )
            needs_hint_reconciliation = _manifest_needs_class_hint_reconciliation(
                existing_manifest,
                class_count_hint,
                objectness_hint,
            )
            if registered_classes_match and not needs_hint_reconciliation:
                parser_plan = resolve_parser_plan(
                    preset_id,
                    output_shape=existing_manifest.output.shape,
                    class_count=existing_manifest.output.class_count,
                    inferred_has_objectness=existing_manifest.output.has_objectness,
                )
                preset_changed = (
                    str(existing_manifest.postprocess.parser_preset) != preset_id
                )
                if preset_changed:
                    existing_manifest = replace(
                        existing_manifest,
                        postprocess=replace(
                            existing_manifest.postprocess,
                            parser_preset=parser_plan.requested_preset,
                        ),
                    )
                if existing_manifest_path == legacy_manifest_path:
                    temporary_path = manifest_path.with_suffix(".json.tmp")
                    try:
                        _write_manifest_preserving_extensions(
                            existing_manifest,
                            temporary_path,
                            source_path=existing_manifest_path,
                        )
                        temporary_path.replace(manifest_path)
                    except Exception:
                        temporary_path.unlink(missing_ok=True)
                        raise
                elif preset_changed:
                    temporary_path = manifest_path.with_suffix(".json.tmp")
                    try:
                        _write_manifest_preserving_extensions(
                            existing_manifest,
                            temporary_path,
                            source_path=existing_manifest_path,
                        )
                        temporary_path.replace(manifest_path)
                    except Exception:
                        temporary_path.unlink(missing_ok=True)
                        raise
                remove_matching_legacy_manifest(path)
                return existing_manifest, preset_changed
        recommendation = recommend_engine_manifest(
            inference,
            artifact_path=path,
            registered_classes=probe_classes,
            registered_input_shape=registered_input_shape,
            parser_preset=preset_id,
        )
        contract = recommendation.contract
        resolved_classes = recommendation.class_names
        parser_plan = recommendation.parser_plan
        output_has_objectness = parser_plan.has_objectness
        if existing_manifest is not None:
            tensor_contract_matches = _manifest_matches_engine_contract(
                existing_manifest,
                contract,
            )
            class_contract_matches = (
                list(existing_manifest.output.class_names) == resolved_classes
                and bool(existing_manifest.output.has_objectness) == output_has_objectness
            )
            parser_preset_matches = (
                str(existing_manifest.postprocess.parser_preset)
                == parser_plan.requested_preset
            )
            needs_hint_reconciliation = _manifest_needs_class_hint_reconciliation(
                existing_manifest,
                class_count_hint,
                objectness_hint,
            )
            if (
                tensor_contract_matches
                and class_contract_matches
                and parser_preset_matches
                and not needs_hint_reconciliation
            ):
                if existing_manifest_path == legacy_manifest_path:
                    temporary_path = manifest_path.with_suffix(".json.tmp")
                    try:
                        _write_manifest_preserving_extensions(
                            existing_manifest,
                            temporary_path,
                            source_path=existing_manifest_path,
                        )
                        temporary_path.replace(manifest_path)
                    except Exception:
                        temporary_path.unlink(missing_ok=True)
                        raise
                remove_matching_legacy_manifest(path)
                return existing_manifest, False
            logger.warning(
                "regenerating DeepStream manifest from TensorRT engine contract "
                "path=%s tensor_contract_matches=%s class_contract_matches=%s "
                "hint_reconciliation=%s manifest_input=%s engine_input=%s",
                path,
                tensor_contract_matches,
                class_contract_matches,
                needs_hint_reconciliation,
                list(existing_manifest.input.shape),
                contract.input_shape,
            )

        template = existing_manifest
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
            runtime_precision=(
                template.runtime.precision if template is not None else runtime_precision
            ),
            input_color_format=(
                template.input.color_format if template is not None else "RGB"
            ),
            input_scale_factor=(
                template.input.scale_factor if template is not None else 1.0 / 255.0
            ),
            maintain_aspect_ratio=(
                template.input.maintain_aspect_ratio if template is not None else False
            ),
            symmetric_padding=(
                template.input.symmetric_padding if template is not None else False
            ),
            output_format=(
                template.output.format
                if template is not None
                else "yolo_cxcywh_class_scores"
            ),
            output_has_objectness=output_has_objectness,
            output_coordinate_mode=(
                template.output.coordinate_mode if template is not None else "pixel"
            ),
            postprocess_parser=(
                template.postprocess.parser if template is not None else "yolo"
            ),
            parser_preset=parser_plan.requested_preset,
            validated=True,
        )
        validate_manifest_engine_artifact(manifest, path)
        temporary_path = manifest_path.with_suffix(".json.tmp")
        try:
            _write_manifest_preserving_extensions(
                manifest,
                temporary_path,
                source_path=existing_manifest_path,
            )
            temporary_path.replace(manifest_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        remove_matching_legacy_manifest(path)
        return manifest, True


def _model_profile_class_names(path: Path) -> list[str]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    profile = raw.get("model_profile") if isinstance(raw, dict) else None
    labels = profile.get("labels") if isinstance(profile, dict) else None
    if not isinstance(labels, list):
        return []
    return [str(item).strip() for item in labels if str(item).strip()]


def remove_matching_legacy_manifest(engine_path: Path) -> bool:
    path = Path(engine_path)
    legacy_manifest_path = path.with_name("model.manifest.json")
    if not legacy_manifest_path.is_file():
        return False
    try:
        legacy_manifest = read_manifest(legacy_manifest_path)
        validate_manifest_engine_artifact(legacy_manifest, path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "preserving unmatched legacy DeepStream manifest path=%s engine=%s error=%s",
            legacy_manifest_path,
            path,
            exc,
        )
        return False
    legacy_manifest_path.unlink(missing_ok=True)
    return True


def _manifest_matches_engine_contract(
    manifest: ModelManifest,
    contract: EngineTensorContract,
) -> bool:
    return (
        manifest.input.name == contract.input_name
        and list(manifest.input.shape) == contract.input_shape
        and normalize_tensor_dtype(manifest.input.dtype) == contract.input_dtype
        and manifest.output.name == contract.output_name
        and list(manifest.output.shape) == contract.output_shape
        and normalize_tensor_dtype(manifest.output.dtype) == contract.output_dtype
    )


def parse_runtime_shape(value: object, label: str) -> list[int]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [item for item in re.split(r"[xX,\s]+", str(value or "").strip()) if item]
    try:
        shape = [int(item) for item in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TensorRT engine probe returned invalid {label}: {value}") from exc
    if not shape or any(item <= 0 for item in shape):
        raise ValueError(f"TensorRT engine probe returned unresolved {label}: {value}")
    return shape


__all__ = [
    "EngineManifestRecommendation",
    "EngineTensorContract",
    "ensure_engine_manifest",
    "remove_matching_legacy_manifest",
    "infer_yolo_output_contract",
    "infer_class_count_hint_from_name",
    "infer_yolo_objectness_hint_from_name",
    "parse_runtime_shape",
    "probe_engine_contract",
    "recommend_engine_manifest",
    "resolve_yolo_class_contract",
]
