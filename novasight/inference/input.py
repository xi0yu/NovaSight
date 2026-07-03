from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TensorInputMode = Literal["gpu_buffer", "cpu_image"]


@dataclass(frozen=True)
class TensorInputShape:
    batch: int
    channels: int
    height: int
    width: int

    def __str__(self) -> str:
        return f"{self.batch}x{self.channels}x{self.height}x{self.width}"


@dataclass(frozen=True)
class PreparedTensorInput:
    mode: TensorInputMode
    buffer: Any
    width: int
    height: int
    pixel_format: str
    source_width: int
    source_height: int
    offset_x: int
    offset_y: int
    needs_resize: bool


def parse_tensor_input_shape(value: str) -> TensorInputShape:
    normalized = value.strip().lower().replace(",", "x").replace(" ", "")
    parts = [part for part in normalized.split("x") if part]
    if len(parts) != 4:
        raise ValueError(f"TensorRT input shape must be NCHW, got: {value}")
    try:
        batch, channels, height, width = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"TensorRT input shape must contain integers: {value}") from exc
    if min(batch, channels, height, width) <= 0:
        raise ValueError(f"TensorRT input shape values must be positive: {value}")
    return TensorInputShape(
        batch=batch,
        channels=channels,
        height=height,
        width=width,
    )


def prepare_tensor_input(frame: Any, shape: TensorInputShape) -> PreparedTensorInput:
    gpu_buffer = getattr(frame, "gpu_buffer", None)
    image = getattr(frame, "image", None)
    if gpu_buffer is not None:
        mode: TensorInputMode = "gpu_buffer"
        buffer = gpu_buffer
    elif image is not None:
        mode = "cpu_image"
        buffer = image
    else:
        raise ValueError("TensorRT input frame has no gpu_buffer or image")

    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    return PreparedTensorInput(
        mode=mode,
        buffer=buffer,
        width=width,
        height=height,
        pixel_format=str(getattr(frame, "pixel_format", "")).upper(),
        source_width=int(getattr(frame, "source_width", width)),
        source_height=int(getattr(frame, "source_height", height)),
        offset_x=int(getattr(frame, "offset_x", 0)),
        offset_y=int(getattr(frame, "offset_y", 0)),
        needs_resize=width != shape.width or height != shape.height,
    )
