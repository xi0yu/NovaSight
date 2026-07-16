from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast


ParserPresetId = Literal[
    "auto",
    "yolov5",
    "yolov8",
    "yolo11",
    "novasight_generic",
]


@dataclass(frozen=True, slots=True)
class ParserPreset:
    preset_id: ParserPresetId
    label: str
    expected_objectness: bool | None


@dataclass(frozen=True, slots=True)
class ParserPlan:
    requested_preset: ParserPresetId
    compatibility: Literal["yolov5", "yolov8_yolo11"]
    has_objectness: bool
    parser_library: str = "novasight_builtin"
    parser_function: str = "NvDsInferParseNovaSight"
    nms_owner: str = "deepstream"

    def asdict(self) -> dict[str, str | bool]:
        return {
            "requested_preset": self.requested_preset,
            "compatibility": self.compatibility,
            "has_objectness": self.has_objectness,
            "parser_library": self.parser_library,
            "parser_function": self.parser_function,
            "nms_owner": self.nms_owner,
        }


PARSER_PRESETS: tuple[ParserPreset, ...] = (
    ParserPreset("auto", "自动识别（推荐）", None),
    ParserPreset("yolov5", "YOLO v5 兼容", True),
    ParserPreset("yolov8", "YOLO v8 兼容", False),
    ParserPreset("yolo11", "YOLO v11 兼容", False),
    ParserPreset("novasight_generic", "NovaSight 通用解析器（内置）", None),
)

_PRESETS_BY_ID = {item.preset_id: item for item in PARSER_PRESETS}
_ALIASES = {
    "": "auto",
    "automatic": "auto",
    "yolo_v5": "yolov5",
    "yolov5_raw": "yolov5",
    "yolo_v8": "yolov8",
    "yolov8_raw": "yolov8",
    "yolo_11": "yolo11",
    "yolo11_raw": "yolo11",
    "generic": "novasight_generic",
    "custom": "novasight_generic",
    "novasight": "novasight_generic",
}


def normalize_parser_preset(value: object) -> ParserPresetId:
    normalized = str(value or "").strip().lower().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _PRESETS_BY_ID:
        supported = ", ".join(item.preset_id for item in PARSER_PRESETS)
        raise ValueError(
            f"unsupported parser preset {value!r}; supported presets: {supported}"
        )
    return cast(ParserPresetId, normalized)


def parser_preset_objectness_hint(value: object) -> bool | None:
    return _PRESETS_BY_ID[normalize_parser_preset(value)].expected_objectness


def resolve_parser_plan(
    preset: object,
    *,
    output_shape: list[int] | tuple[int, ...],
    class_count: int,
    inferred_has_objectness: bool,
) -> ParserPlan:
    preset_id = normalize_parser_preset(preset)
    definition = _PRESETS_BY_ID[preset_id]
    count = int(class_count)
    if count <= 0:
        raise ValueError("parser preset resolution requires a positive class_count")
    dimensions = [int(value) for value in output_shape]
    if len(dimensions) == 3 and dimensions[0] == 1:
        dimensions = dimensions[1:]
    if len(dimensions) != 2:
        raise ValueError(
            "NovaSight built-in YOLO parser requires [C,N], [N,C], [1,C,N], or [1,N,C]"
        )

    expected_objectness = definition.expected_objectness
    has_objectness = (
        bool(inferred_has_objectness)
        if expected_objectness is None
        else expected_objectness
    )
    if expected_objectness is not None and bool(inferred_has_objectness) != expected_objectness:
        raise ValueError(
            f"{definition.label} conflicts with the detected objectness contract "
            f"(detected has_objectness={str(bool(inferred_has_objectness)).lower()})"
        )
    expected_channels = count + (5 if has_objectness else 4)
    if expected_channels not in dimensions:
        raise ValueError(
            f"{definition.label} expects {expected_channels} output channels for "
            f"{count} classes, got shape={list(output_shape)}"
        )
    candidates = max(dimensions)
    if candidates <= expected_channels:
        raise ValueError(
            "NovaSight built-in YOLO parser requires the candidate dimension to exceed channels"
        )
    return ParserPlan(
        requested_preset=preset_id,
        compatibility="yolov5" if has_objectness else "yolov8_yolo11",
        has_objectness=has_objectness,
    )


def parser_preset_payload() -> list[dict[str, str | bool | None]]:
    return [
        {
            "id": item.preset_id,
            "label": item.label,
            "expected_objectness": item.expected_objectness,
            "external_library": False,
        }
        for item in PARSER_PRESETS
    ]


__all__ = [
    "PARSER_PRESETS",
    "ParserPlan",
    "ParserPreset",
    "ParserPresetId",
    "normalize_parser_preset",
    "parser_preset_objectness_hint",
    "parser_preset_payload",
    "resolve_parser_plan",
]
