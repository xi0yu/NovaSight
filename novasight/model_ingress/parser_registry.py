from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParserDefinition:
    parser_type: str
    runtime_decoder: str
    built_in_nms: bool


PARSER_REGISTRY: dict[str, ParserDefinition] = {
    "yolov5_raw": ParserDefinition("yolov5_raw", "yolov5", False),
    "yolov8_raw": ParserDefinition("yolov8_raw", "yolov8", False),
    "yolo11_raw": ParserDefinition("yolo11_raw", "yolov8", False),
    # Rockchip RKNN/YOLOv5 exports three raw NCHW heads.  It is decoded by the
    # DeepStream native parser; it is intentionally not offered to the Python
    # TensorRT fallback.
    "rockchip_yolov5": ParserDefinition("rockchip_yolov5", "rockchip_yolov5", False),
}


def get_parser_definition(parser_type: str) -> ParserDefinition:
    key = str(parser_type).strip().lower().replace("-", "_")
    try:
        return PARSER_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(PARSER_REGISTRY))
        raise ValueError(
            f"unsupported NovaSight parser {parser_type!r}; supported parsers: {supported}"
        ) from exc


__all__ = ["PARSER_REGISTRY", "ParserDefinition", "get_parser_definition"]
