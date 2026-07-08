from __future__ import annotations

from novasight.inference.decoders.base import BaseDecoder
from novasight.inference.decoders.yolo_v5 import YoloV5Decoder
from novasight.inference.decoders.yolo_v8 import YoloV8Decoder, YoloV8EndToEndDecoder


PARSER_REGISTRY: dict[str, type[BaseDecoder]] = {
    "yolov5": YoloV5Decoder,
    "yolov8": YoloV8Decoder,
    "yolo": YoloV8Decoder,
    "yolov8_e2e": YoloV8EndToEndDecoder,
    "yolov8_efficientnms": YoloV8EndToEndDecoder,
}


def get_decoder_class(parser_name: str) -> type[BaseDecoder]:
    key = parser_name.strip().lower().replace("-", "_")
    try:
        return PARSER_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(PARSER_REGISTRY))
        raise ValueError(f"unsupported parser {parser_name!r}; supported parsers: {supported}") from exc


def create_decoder(parser_name: str) -> BaseDecoder:
    return get_decoder_class(parser_name)()
