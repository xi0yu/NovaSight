from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TensorInputMode = Literal["gpu_buffer", "host_frame"]
TENSOR_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
}
_TENSOR_DTYPE_ALIASES = {
    "fp32": "float32",
    "float": "float32",
    "f32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "half": "float16",
    "f16": "float16",
    "float16": "float16",
}


@dataclass(frozen=True)
class TensorInputShape:
    batch: int
    channels: int
    height: int
    width: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", normalize_tensor_dtype(self.dtype))

    def __str__(self) -> str:
        return f"{self.batch}x{self.channels}x{self.height}x{self.width}"

    @property
    def nbytes(self) -> int:
        return (
            int(self.batch)
            * int(self.channels)
            * int(self.height)
            * int(self.width)
            * tensor_dtype_size_bytes(self.dtype)
        )


def normalize_tensor_dtype(value: str | Any) -> str:
    normalized = str(value or "float32").strip().lower().replace("-", "").replace("_", "")
    dtype = _TENSOR_DTYPE_ALIASES.get(normalized)
    if dtype is None:
        accepted = ", ".join(sorted(TENSOR_DTYPE_BYTES))
        raise ValueError(f"unsupported tensor dtype {value!r}; expected one of: {accepted}")
    return dtype


def tensor_dtype_size_bytes(value: str | Any) -> int:
    return TENSOR_DTYPE_BYTES[normalize_tensor_dtype(value)]


@dataclass(frozen=True)
class PreparedTensorInput:
    mode: TensorInputMode
    buffer: Any
    frame_id: int
    capture_ts_ns: int
    width: int
    height: int
    pixel_format: str
    source_width: int
    source_height: int
    offset_x: int
    offset_y: int
    needs_resize: bool
    resource_kind: str = ""
    resource_memory: str = ""
    resource_source: str = ""
    dmabuf_fd: int | None = None
    resource_metadata: dict[str, Any] = field(default_factory=dict)
    resource_width: int = 0
    resource_height: int = 0
    resource_pixel_format: str = ""

    @property
    def gst_buffer_ptr(self) -> int | None:
        return _optional_positive_int(self.resource_metadata.get("gst_buffer_ptr"))


def parse_tensor_input_shape(value: str) -> TensorInputShape:
    normalized = (
        value.strip()
        .lower()
        .replace(",", "x")
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )
    parts = [part for part in normalized.split("x") if part]
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"TensorRT input shape must contain integers: {value}") from exc
    if len(numbers) == 2:
        height, width = numbers
        batch = 1
        channels = 3
    elif len(numbers) == 3:
        channels, height, width = numbers
        batch = 1
    elif len(numbers) == 4:
        batch, channels, height, width = numbers
    else:
        raise ValueError(
            f"TensorRT input shape must be NCHW, CHW, or HW, got: {value}"
        )
    if min(batch, channels, height, width) <= 0:
        raise ValueError(f"TensorRT input shape values must be positive: {value}")
    if channels not in {1, 3, 4}:
        raise ValueError(f"TensorRT input channels must be 1, 3, or 4, got: {channels}")
    return TensorInputShape(
        batch=batch,
        channels=channels,
        height=height,
        width=width,
    )


def prepare_tensor_input(frame: Any, shape: TensorInputShape) -> PreparedTensorInput:
    frame_resource = getattr(frame, "frame_resource", None)
    gpu_buffer = getattr(frame, "gpu_buffer", None)
    if _resource_gpu_accessible(frame_resource):
        mode: TensorInputMode = "gpu_buffer"
        buffer = getattr(frame_resource, "handle")
        resource_kind = str(getattr(frame_resource, "kind", "") or "")
        resource_memory = str(getattr(frame_resource, "memory", "") or "")
        resource_source = str(getattr(frame_resource, "source", "") or "")
        dmabuf_fd = _optional_int_attr(frame_resource, "dmabuf_fd")
        resource_metadata = _dict_attr(frame_resource, "metadata")
        resource_width = _int_attr(frame_resource, "width", 0)
        resource_height = _int_attr(frame_resource, "height", 0)
        resource_pixel_format = str(
            getattr(frame_resource, "pixel_format", "") or ""
        ).upper()
    elif gpu_buffer is not None:
        mode = "gpu_buffer"
        buffer = gpu_buffer
        resource_kind = str(getattr(frame_resource, "kind", "") or "external_gpu_buffer")
        resource_memory = str(getattr(frame_resource, "memory", "") or "unknown")
        resource_source = str(getattr(frame_resource, "source", "") or "")
        dmabuf_fd = _optional_int_attr(frame_resource, "dmabuf_fd")
        resource_metadata = _dict_attr(frame_resource, "metadata")
        resource_width = _int_attr(frame_resource, "width", 0)
        resource_height = _int_attr(frame_resource, "height", 0)
        resource_pixel_format = str(
            getattr(frame_resource, "pixel_format", "") or ""
        ).upper()
    elif getattr(frame, "image", None) is not None:
        mode = "host_frame"
        buffer = getattr(frame, "image")
        resource_kind = str(getattr(frame_resource, "kind", "") or "host_frame")
        resource_memory = str(getattr(frame_resource, "memory", "") or "cpu")
        resource_source = str(
            getattr(frame_resource, "source", "") or getattr(frame, "source_backend", "")
        )
        dmabuf_fd = None
        resource_metadata = _dict_attr(frame_resource, "metadata")
        resource_width = _int_attr(frame_resource, "width", _int_attr(frame, "width", 0))
        resource_height = _int_attr(frame_resource, "height", _int_attr(frame, "height", 0))
        resource_pixel_format = str(
            getattr(frame_resource, "pixel_format", "") or getattr(frame, "pixel_format", "")
        ).upper()
    else:
        raise ValueError("TensorRT input frame has no GPU/NVMM resource or CPU-readable image")

    width = _int_attr(frame, "width", 1)
    height = _int_attr(frame, "height", 1)
    pixel_format = str(getattr(frame, "pixel_format", "") or "").upper()
    if not pixel_format and frame_resource is not None:
        pixel_format = str(getattr(frame_resource, "pixel_format", "") or "").upper()
    return PreparedTensorInput(
        mode=mode,
        buffer=buffer,
        frame_id=_int_attr(frame, "frame_id", 0),
        capture_ts_ns=_int_attr(
            frame,
            "capture_ts_ns",
            _int_attr(frame, "ts_ns", 0),
        ),
        width=width,
        height=height,
        pixel_format=pixel_format,
        source_width=_int_attr(
            frame,
            "capture_width",
            _int_attr(frame, "source_width", width),
        ),
        source_height=_int_attr(
            frame,
            "capture_height",
            _int_attr(frame, "source_height", height),
        ),
        offset_x=_int_attr(
            frame,
            "roi_x",
            _int_attr(frame, "offset_x", _int_attr(frame, "roi_offset_x", 0)),
        ),
        offset_y=_int_attr(
            frame,
            "roi_y",
            _int_attr(frame, "offset_y", _int_attr(frame, "roi_offset_y", 0)),
        ),
        needs_resize=width != shape.width or height != shape.height,
        resource_kind=resource_kind,
        resource_memory=resource_memory,
        resource_source=resource_source,
        dmabuf_fd=dmabuf_fd,
        resource_metadata=resource_metadata,
        resource_width=resource_width,
        resource_height=resource_height,
        resource_pixel_format=resource_pixel_format,
    )


def _resource_gpu_accessible(resource: Any) -> bool:
    if resource is None:
        return False
    value = getattr(resource, "gpu_accessible", False)
    if isinstance(value, bool):
        return value
    memory = str(getattr(resource, "memory", "") or "").lower()
    return memory in {"nvmm", "dmabuf", "cuda"}


def _int_attr(obj: object, name: str, default: int) -> int:
    value = getattr(obj, name, None)
    if value is None:
        return int(default)
    return int(value)


def _optional_int_attr(obj: object | None, name: str) -> int | None:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        result = int(value)
    except Exception:
        return None
    return result if result >= 0 else None


def _optional_positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except Exception:
        return None
    return result if result > 0 else None


def _dict_attr(obj: object | None, name: str) -> dict[str, Any]:
    if obj is None:
        return {}
    value = getattr(obj, name, None)
    if not isinstance(value, dict):
        return {}
    return dict(value)
