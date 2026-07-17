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
    resolve_decoded_nms_parser_plan,
    resolve_efficient_nms_parser_plan,
    resolve_rockchip_yolov5_parser_plan,
    resolve_parser_plan,
)


_MANIFEST_LOCK = threading.RLock()
logger = logging.getLogger("novasight.deepstream.model_manifest")
_ROCKCHIP_YOLOV5_STRIDES = (8, 16, 32)
_ROCKCHIP_YOLOV5_ANCHORS = (
    (10.0, 13.0, 16.0, 30.0, 33.0, 23.0),
    (30.0, 61.0, 62.0, 45.0, 59.0, 119.0),
    (116.0, 90.0, 156.0, 198.0, 373.0, 326.0),
)


@dataclass(frozen=True, slots=True)
class EngineTensorContract:
    input_name: str
    input_shape: list[int]
    input_dtype: str
    output_name: str
    output_shape: list[int]
    output_dtype: str
    output_format: str = "yolo_cxcywh_class_scores"
    postprocess_parser: str = "yolo"
    output_bindings: tuple[TensorSpec, ...] = ()
    output_strides: tuple[int, ...] = ()
    output_anchors: tuple[tuple[float, ...], ...] = ()
    inferred_class_count: int | None = None


@dataclass(frozen=True, slots=True)
class EngineManifestRecommendation:
    contract: EngineTensorContract
    class_names: list[str]
    output_has_objectness: bool
    parser_plan: ParserPlan


@dataclass(frozen=True, slots=True)
class PreparedEngineManifest:
    """One-shot proof that an Engine manifest was validated in this operation."""

    engine_path: Path
    engine_signature: tuple[int, int, int, int]
    manifest: ModelManifest

    @classmethod
    def create(
        cls,
        *,
        engine_path: Path,
        manifest: ModelManifest,
    ) -> "PreparedEngineManifest":
        path = Path(engine_path).resolve(strict=False)
        return cls(
            engine_path=path,
            engine_signature=_engine_file_signature(path),
            manifest=manifest,
        )

    def take_if_current(self, engine_path: Path) -> ModelManifest | None:
        path = Path(engine_path).resolve(strict=False)
        if path != self.engine_path:
            return None
        try:
            current_signature = _engine_file_signature(path)
        except OSError:
            return None
        if current_signature != self.engine_signature:
            return None
        return self.manifest


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
    class_count_hint: int | None = None,
) -> EngineTensorContract:
    probe = getattr(inference, "probe", None)
    if not callable(probe):
        raise ValueError(
            "TensorRT engine probe is unavailable; refusing to guess DeepStream tensor bindings"
        )
    status = dict(probe(artifact_path, classes, registered_input_shape))
    io_tensors = status.get("io_tensors")
    contract_only_probe = (
        isinstance(io_tensors, list)
        and "unsupported TensorRT detection output contract"
        in str(status.get("reason") or "")
    )
    if status.get("loaded") is not True and not contract_only_probe:
        raise ValueError(
            f"TensorRT engine probe failed: {status.get('reason') or 'loaded=false'}"
        )
    if isinstance(io_tensors, list):
        return _contract_from_io_tensors(
            io_tensors,
            class_count_hint=class_count_hint,
        )
    outputs = status.get("outputs")
    input_name = str(status.get("input_name") or "").strip()
    selected_output: dict[str, object] | None = None
    efficient_nms: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ] | None = None
    if isinstance(outputs, dict):
        output_tensors = [
            {
                "name": str(name),
                **(dict(value) if isinstance(value, dict) else {}),
            }
            for name, value in outputs.items()
        ]
        efficient_nms = _efficient_nms_bindings(output_tensors)
        selected_output = (
            efficient_nms[1]
            if efficient_nms is not None
            else _select_raw_yolo_output(
                output_tensors,
                class_count_hint=class_count_hint,
            )
        )
    output_name = str(
        (selected_output or {}).get("name") or status.get("output_name") or ""
    ).strip()
    if not input_name or not output_name:
        raise ValueError("TensorRT engine probe did not expose input/output tensor names")
    return EngineTensorContract(
        input_name=input_name,
        input_shape=parse_runtime_shape(status.get("input_shape"), "input_shape"),
        input_dtype=normalize_tensor_dtype(status.get("input_dtype")),
        output_name=output_name,
        output_shape=parse_runtime_shape(
            (selected_output or {}).get("shape", status.get("output_shape")),
            "output_shape",
        ),
        output_dtype=normalize_tensor_dtype(
            (selected_output or {}).get("dtype", status.get("output_dtype"))
        ),
        output_format=(
            "efficientnms_boxes_scores_classes"
            if efficient_nms is not None
            else "yolo_cxcywh_class_scores"
        ),
        postprocess_parser="efficientnms" if efficient_nms is not None else "yolo",
        output_bindings=tuple(
            TensorSpec(
                name=str(item.get("name") or ""),
                shape=parse_runtime_shape(item.get("shape"), "output binding shape"),
                dtype=_normalize_output_binding_dtype(item.get("dtype")),
                layout="NCHW",
            )
            for item in (efficient_nms or [])
        ),
    )


def _contract_from_io_tensors(
    io_tensors: list[object],
    *,
    class_count_hint: int | None = None,
) -> EngineTensorContract:
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
    if len(inputs) != 1:
        raise ValueError(
            "DeepStream requires exactly one TensorRT image input, "
            f"got inputs={len(inputs)} outputs={len(outputs)}"
        )
    input_tensor = inputs[0]
    input_shape = parse_runtime_shape(input_tensor.get("shape"), "input_shape")
    rockchip_yolov5 = _rockchip_yolov5_bindings(
        outputs,
        input_shape=input_shape,
        class_count_hint=class_count_hint,
    )
    efficient_nms = _efficient_nms_bindings(outputs) if rockchip_yolov5 is None else None
    output_tensor = (
        rockchip_yolov5[0][0]
        if rockchip_yolov5 is not None
        else (
            efficient_nms[1]
            if efficient_nms is not None
            else _select_raw_yolo_output(outputs, class_count_hint=class_count_hint)
        )
    )
    decoded_nms = False
    if len(outputs) == 1:
        shape = parse_runtime_shape(outputs[0].get("shape"), "output shape")
        # [1,N,6] is ambiguous: a small N is the usual end-to-end NMS top-k,
        # while a large N is also the canonical one-class YOLOv5 raw tensor
        # (cx,cy,w,h,obj,class_score). Never classify the dense raw form as NMS.
        decoded_nms = (
            len(shape) == 3
            and shape[0] == 1
            and shape[2] == 6
            and 6 < shape[1] <= 512
        )
    input_name = str(input_tensor.get("name") or "").strip()
    output_name = str(output_tensor.get("name") or "").strip()
    if not input_name or not output_name:
        raise ValueError("TensorRT engine probe returned an unnamed I/O tensor")
    return EngineTensorContract(
        input_name=input_name,
        input_shape=input_shape,
        input_dtype=normalize_tensor_dtype(input_tensor.get("dtype")),
        output_name=output_name,
        output_shape=parse_runtime_shape(output_tensor.get("shape"), "output_shape"),
        output_dtype=normalize_tensor_dtype(output_tensor.get("dtype")),
        output_format=(
            "decoded_boxes6" if decoded_nms else (
            "rockchip_yolov5_three_scale"
            if rockchip_yolov5 is not None
            else (
                "efficientnms_boxes_scores_classes"
                if efficient_nms is not None
                else "yolo_cxcywh_class_scores"
            ))
        ),
        postprocess_parser=(
            "decoded_nms" if decoded_nms else (
            "rockchip_yolov5"
            if rockchip_yolov5 is not None
            else ("efficientnms" if efficient_nms is not None else "yolo")
        )),
        output_bindings=tuple(
            TensorSpec(
                name=str(item.get("name") or ""),
                shape=parse_runtime_shape(item.get("shape"), "output binding shape"),
                dtype=_normalize_output_binding_dtype(item.get("dtype")),
                layout="NCHW",
            )
            for item in (
                [entry[0] for entry in rockchip_yolov5]
                if rockchip_yolov5 is not None
                else (efficient_nms or [])
            )
        ),
        output_strides=(
            _ROCKCHIP_YOLOV5_STRIDES if rockchip_yolov5 is not None else ()
        ),
        output_anchors=(
            _ROCKCHIP_YOLOV5_ANCHORS if rockchip_yolov5 is not None else ()
        ),
        inferred_class_count=(
            rockchip_yolov5[0][1] if rockchip_yolov5 is not None else None
        ),
    )


def _rockchip_yolov5_bindings(
    outputs: list[dict[str, object]],
    *,
    input_shape: list[int],
    class_count_hint: int | None,
) -> tuple[tuple[dict[str, object], int], ...] | None:
    if len(outputs) != 3 or len(input_shape) != 4 or input_shape[2] != input_shape[3]:
        return None
    described: list[tuple[dict[str, object], int, int]] = []
    channels: int | None = None
    for output in outputs:
        try:
            shape = parse_runtime_shape(
                output.get("shape"),
                f"output {output.get('name') or '<unnamed>'}",
            )
        except ValueError:
            return None
        if len(shape) != 4 or shape[0] != 1 or shape[2] != shape[3]:
            return None
        if normalize_tensor_dtype(output.get("dtype")) not in {"float16", "float32"}:
            return None
        channels = shape[1] if channels is None else channels
        if shape[1] != channels:
            return None
        described.append((output, shape[2], shape[1]))
    described.sort(key=lambda item: item[1], reverse=True)
    grids = [item[1] for item in described]
    if grids[1] * 2 != grids[0] or grids[2] * 2 != grids[1]:
        return None
    strides = [int(input_shape[2]) // grid for grid in grids]
    if strides != list(_ROCKCHIP_YOLOV5_STRIDES):
        return None
    if channels is None or channels % 3 != 0:
        return None
    inferred_class_count = channels // 3 - 5
    if inferred_class_count <= 0:
        return None
    if class_count_hint is not None and int(class_count_hint) != inferred_class_count:
        raise ValueError(
            "Rockchip YOLOv5 output class count conflicts with category configuration "
            f"(heads infer {inferred_class_count}, configured {class_count_hint})"
        )
    return tuple((item[0], inferred_class_count) for item in described)


def _efficient_nms_bindings(
    outputs: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]] | None:
    if len(outputs) != 4:
        return None
    described: list[tuple[dict[str, object], str, list[int], str]] = []
    for output in outputs:
        name = str(output.get("name") or "").strip().lower()
        try:
            shape = parse_runtime_shape(output.get("shape"), f"output {name}")
        except ValueError:
            return None
        dtype = _normalize_output_binding_dtype(output.get("dtype"))
        described.append((output, name, shape, dtype))

    def named(tokens: tuple[str, ...]) -> list[tuple[dict[str, object], str, list[int], str]]:
        return [item for item in described if any(token in item[1] for token in tokens)]

    count_candidates = named(("num_det", "numdet", "count", "keep_count"))
    boxes_candidates = named(("box", "bbox"))
    scores_candidates = named(("score", "conf"))
    classes_candidates = named(("class", "label"))
    if not all(
        len(items) == 1
        for items in (
            count_candidates,
            boxes_candidates,
            scores_candidates,
            classes_candidates,
        )
    ):
        return None
    count = count_candidates[0]
    boxes = boxes_candidates[0]
    scores = scores_candidates[0]
    classes = classes_candidates[0]
    boxes_shape = boxes[2]
    box_count = boxes_shape[-2] if len(boxes_shape) >= 2 and boxes_shape[-1] == 4 else 0
    if box_count <= 0 or _element_count(count[2]) != 1:
        return None
    if _element_count(scores[2]) != box_count or _element_count(classes[2]) != box_count:
        return None
    return (count[0], boxes[0], scores[0], classes[0])


def _element_count(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _normalize_output_binding_dtype(value: object) -> str:
    normalized = str(value or "float32").strip().lower().replace("_", "")
    aliases = {
        "float": "float32",
        "fp32": "float32",
        "float32": "float32",
        "half": "float16",
        "fp16": "float16",
        "float16": "float16",
        "int32": "int32",
        "int": "int32",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported TensorRT output binding dtype {value!r}")
    return aliases[normalized]


def _select_raw_yolo_output(
    outputs: list[dict[str, object]],
    *,
    class_count_hint: int | None,
) -> dict[str, object]:
    if not outputs:
        raise ValueError("TensorRT engine does not expose an output tensor")
    if len(outputs) == 1:
        return outputs[0]
    expected_channels = (
        {4 + int(class_count_hint), 5 + int(class_count_hint)}
        if class_count_hint is not None and int(class_count_hint) > 0
        else set()
    )
    candidates: list[dict[str, object]] = []
    descriptions: list[str] = []
    for output in outputs:
        name = str(output.get("name") or "<unnamed>")
        try:
            shape = parse_runtime_shape(output.get("shape"), f"output {name}")
        except ValueError:
            shape = []
        descriptions.append(f"{name}:{shape or '<unknown>'}")
        dimensions = shape[1:] if len(shape) == 3 and shape[0] == 1 else shape
        raw_shape_candidate = (
            len(dimensions) == 2
            and min(dimensions) > 4
            and max(dimensions) > min(dimensions)
            and not (6 in dimensions and max(dimensions) <= 512)
        )
        channel_contract_matches = (
            any(value in expected_channels for value in dimensions)
            if expected_channels
            else raw_shape_candidate
        )
        if raw_shape_candidate and channel_contract_matches:
            candidates.append(output)
    if len(candidates) == 1:
        return candidates[0]
    detail = ", ".join(descriptions)
    if not candidates:
        raise ValueError(
            "DeepStream could not identify a supported detection contract among multiple "
            f"TensorRT outputs ({detail}). Expected one raw YOLO tensor or named "
            "num_dets/boxes/scores/classes EfficientNMS outputs."
        )
    raise ValueError(
        "DeepStream found multiple possible raw YOLO detection tensors; refusing an "
        f"ambiguous binding selection ({detail})"
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
        class_count_hint=(
            infer_class_count_hint_from_name(path.name)
            or (
                None
                if _automatic_class_names(registered_classes)
                else len(registered_classes) or None
            )
        ),
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
    if contract.postprocess_parser == "rockchip_yolov5":
        inferred_count = int(contract.inferred_class_count or 0)
        if inferred_count <= 0:
            raise ValueError("Rockchip YOLOv5 contract did not expose a class count")
        rockchip_classes = list(registered_classes)
        if _automatic_class_names(rockchip_classes):
            rockchip_classes = [f"class_{index}" for index in range(inferred_count)]
        elif len(rockchip_classes) != inferred_count:
            raise ValueError(
                "Rockchip YOLOv5 output class count conflicts with category names "
                f"(heads infer {inferred_count}, configured {len(rockchip_classes)})"
            )
        parser_plan = resolve_rockchip_yolov5_parser_plan(preset_id)
        return EngineManifestRecommendation(
            contract=contract,
            class_names=rockchip_classes,
            output_has_objectness=True,
            parser_plan=parser_plan,
        )
    if contract.postprocess_parser in {"efficientnms", "decoded_nms"}:
        efficient_classes = list(registered_classes)
        if _automatic_class_names(efficient_classes):
            inferred_count = (
                infer_class_count_hint_from_name(path.name)
                or (len(efficient_classes) if len(efficient_classes) > 1 else 0)
            )
            if inferred_count <= 0:
                raise ValueError(
                    "EfficientNMS 输出本身不包含类别总数，无法安全自动生成类别映射。"
                    "请先在项目的类别配置中填写完整类别名称和顺序（例如：敌人、倒地、队友），"
                    "然后重新发布模型；仅有默认类别 target 时系统不会猜测类别数量。"
                )
            efficient_classes = [f"class_{index}" for index in range(inferred_count)]
        parser_plan = (
            resolve_decoded_nms_parser_plan(preset_id)
            if contract.postprocess_parser == "decoded_nms"
            else resolve_efficient_nms_parser_plan(preset_id)
        )
        return EngineManifestRecommendation(
            contract=contract,
            class_names=efficient_classes,
            output_has_objectness=False,
            parser_plan=parser_plan,
        )
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
    if str(manifest.postprocess.parser).strip().lower() != "yolo":
        return False
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


def _engine_file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


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
                parser_plan = _resolve_manifest_parser_plan(
                    existing_manifest,
                    preset_id=preset_id,
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
        engine_signature_before = _engine_file_signature(path)
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
            output_format=contract.output_format,
            output_has_objectness=output_has_objectness,
            output_coordinate_mode=(
                template.output.coordinate_mode if template is not None else "pixel"
            ),
            postprocess_parser=contract.postprocess_parser,
            parser_preset=parser_plan.requested_preset,
            output_bindings=list(contract.output_bindings),
            output_strides=list(contract.output_strides),
            output_anchors=[list(scale) for scale in contract.output_anchors],
            validated=True,
        )
        engine_signature_after = _engine_file_signature(path)
        if engine_signature_after != engine_signature_before:
            raise ValueError("TensorRT engine changed while its manifest was being generated")
        if int(manifest.artifact.size_bytes) != engine_signature_after[2]:
            raise ValueError("generated manifest does not match TensorRT engine size")
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
    manifest_bindings = [
        (item.name, list(item.shape), _normalize_output_binding_dtype(item.dtype))
        for item in manifest.output.bindings
    ]
    contract_bindings = [
        (item.name, list(item.shape), _normalize_output_binding_dtype(item.dtype))
        for item in contract.output_bindings
    ]
    return (
        manifest.input.name == contract.input_name
        and list(manifest.input.shape) == contract.input_shape
        and normalize_tensor_dtype(manifest.input.dtype) == contract.input_dtype
        and manifest.output.name == contract.output_name
        and list(manifest.output.shape) == contract.output_shape
        and normalize_tensor_dtype(manifest.output.dtype) == contract.output_dtype
        and str(manifest.output.format).strip().lower() == contract.output_format
        and str(manifest.postprocess.parser).strip().lower()
        == contract.postprocess_parser
        and manifest_bindings == contract_bindings
        and tuple(int(value) for value in manifest.output.strides)
        == contract.output_strides
        and tuple(
            tuple(float(value) for value in scale)
            for scale in manifest.output.anchors
        )
        == contract.output_anchors
    )


def _resolve_manifest_parser_plan(
    manifest: ModelManifest,
    *,
    preset_id: object,
) -> ParserPlan:
    parser = str(manifest.postprocess.parser).strip().lower()
    if parser == "decoded_nms":
        return resolve_decoded_nms_parser_plan(preset_id)
    if parser == "efficientnms":
        return resolve_efficient_nms_parser_plan(preset_id)
    if parser == "rockchip_yolov5":
        return resolve_rockchip_yolov5_parser_plan(preset_id)
    return resolve_parser_plan(
        preset_id,
        output_shape=manifest.output.shape,
        class_count=manifest.output.class_count,
        inferred_has_objectness=manifest.output.has_objectness,
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
