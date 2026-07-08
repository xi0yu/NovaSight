from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from novasight.inference.contracts import InferenceDetection


class BaseDecoder(ABC):
    """Compatibility decoder for tensor outputs that are not parsed inside nvinfer."""

    parser_name: str = ""
    requires_cpu_postprocess: bool = True
    nms_strategy: str = "python"

    @abstractmethod
    def decode(
        self,
        layer_outputs: list[Any],
        *,
        confidence_threshold: float,
        nms_threshold: float,
        class_count: int | None = None,
        debug: dict[str, Any] | None = None,
    ) -> list[InferenceDetection]:
        """Decode nvinfer/engine layer outputs into normalized inference detections."""


def single_output(layer_outputs: list[Any], *, parser_name: str) -> Any:
    if not layer_outputs:
        raise ValueError(f"{parser_name} decoder requires at least one output layer")
    if len(layer_outputs) != 1:
        raise ValueError(f"{parser_name} decoder expects one output layer, got {len(layer_outputs)}")
    return layer_outputs[0]
