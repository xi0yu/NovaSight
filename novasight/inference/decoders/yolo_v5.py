from __future__ import annotations

from typing import Any

from novasight.inference.contracts import InferenceDetection
from novasight.inference.decoders.base import BaseDecoder, single_output
from novasight.inference.postprocess.yolo import decode_nx6_detections


class YoloV5Decoder(BaseDecoder):
    parser_name = "yolov5"
    requires_cpu_postprocess = True
    nms_strategy = "python-class-aware"

    def decode(
        self,
        layer_outputs: list[Any],
        *,
        confidence_threshold: float,
        nms_threshold: float,
        class_count: int | None = None,
        debug: dict[str, Any] | None = None,
    ) -> list[InferenceDetection]:
        if debug is not None:
            debug.update(
                {
                    "parser": self.parser_name,
                    "requires_cpu_postprocess": self.requires_cpu_postprocess,
                    "nms_strategy": self.nms_strategy,
                }
            )
        return decode_nx6_detections(
            single_output(layer_outputs, parser_name=self.parser_name),
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            class_count=class_count,
            debug=debug,
        )
