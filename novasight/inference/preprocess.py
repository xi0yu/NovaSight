from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .input import PreparedTensorInput, TensorInputShape, normalize_tensor_dtype


GPU_RESOURCE_PREPROCESS_NOT_IMPLEMENTED = "GPU_RESOURCE_PREPROCESS_NOT_IMPLEMENTED"


class TensorPreprocessError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class DeviceTensor:
    device_ptr: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str = "float32"
    stream: Any | None = None
    owner: Any | None = None


@dataclass(frozen=True)
class TensorPreprocessResult:
    tensor: Any
    backend: str
    input_mode: str
    resource_kind: str
    resource_memory: str
    location: str = "host"
    zero_copy: bool = False
    reason: str = ""
    timings: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None

    def debug_payload(self) -> dict[str, Any]:
        payload = {
            "preprocess_backend": self.backend,
            "preprocess_location": self.location,
            "preprocess_zero_copy": self.zero_copy,
            "preprocess_reason": self.reason,
        }
        if self.timings:
            payload["preprocess_native_timings"] = dict(self.timings)
        if self.metadata:
            payload.update(dict(self.metadata))
        return payload


class GpuResourcePreprocessor(Protocol):
    def prepare(
        self,
        prepared: PreparedTensorInput,
        shape: TensorInputShape,
    ) -> TensorPreprocessResult: ...


class CpuPreprocessor:
    backend = "cpu"

    def prepare(
        self,
        prepared: PreparedTensorInput,
        shape: TensorInputShape,
        *,
        model_preprocess: Any | None = None,
    ) -> TensorPreprocessResult:
        return prepare_host_tensor(
            prepared,
            shape,
            model_preprocess=model_preprocess,
        )

    def status(self) -> dict[str, Any]:
        return {
            "selected": self.backend,
            "available": True,
            "location": "host",
            "reason": "temporary CPU host bridge before TensorRT CUDA upload",
        }


def prepare_tensor(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    gpu_preprocessor: GpuResourcePreprocessor | None = None,
    model_preprocess: Any | None = None,
) -> TensorPreprocessResult:
    if prepared.mode == "host_frame":
        return prepare_host_tensor(
            prepared,
            shape,
            model_preprocess=model_preprocess,
        )
    if prepared.mode != "gpu_buffer" or gpu_preprocessor is None:
        raise_gpu_resource_preprocess_not_implemented(prepared)
    configure = getattr(gpu_preprocessor, "configure_model_preprocess", None)
    if callable(configure):
        configure(model_preprocess)
    result = gpu_preprocessor.prepare(prepared, shape)
    _validate_device_preprocess_result(result, shape)
    return result


def prepare_host_tensor(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    model_preprocess: Any | None = None,
) -> TensorPreprocessResult:
    if prepared.mode != "host_frame":
        raise TensorPreprocessError(
            GPU_RESOURCE_PREPROCESS_NOT_IMPLEMENTED,
            f"host tensor preprocess expects host_frame input, got {prepared.mode}",
        )
    if int(shape.batch) != 1 or int(shape.channels) != 3:
        raise TensorPreprocessError(
            "CPU_PREPROCESS_UNSUPPORTED_SHAPE",
            f"CPU preprocess supports N=1,C=3 tensors only, got {shape}",
        )
    import time

    import numpy as np

    total_start_ns = time.monotonic_ns()
    convert_start_ns = total_start_ns
    rgb = _host_frame_to_rgb(prepared, np=np)
    convert_done_ns = time.monotonic_ns()
    resize_start_ns = convert_done_ns
    color_format = str(
        getattr(model_preprocess, "color_format", "RGB") or "RGB"
    ).upper()
    if color_format not in {"RGB", "BGR"}:
        raise TensorPreprocessError(
            "CPU_PREPROCESS_UNSUPPORTED_COLOR_FORMAT",
            f"CPU preprocess supports RGB or BGR model input, got {color_format}",
        )
    model_pixels = rgb if color_format == "RGB" else np.ascontiguousarray(rgb[:, :, ::-1])
    model_pixels, resize_metadata = _resize_for_profile(
        model_pixels,
        width=int(shape.width),
        height=int(shape.height),
        np=np,
        model_preprocess=model_preprocess,
    )
    resize_done_ns = time.monotonic_ns()
    layout_start_ns = resize_done_ns
    dtype = np.float16 if normalize_tensor_dtype(shape.dtype) == "float16" else np.float32
    tensor = np.ascontiguousarray(
        model_pixels.transpose(2, 0, 1)[None, :, :, :],
        dtype=dtype,
    )
    if np.issubdtype(tensor.dtype, np.floating):
        scale = float(getattr(model_preprocess, "scale", 1.0 / 255.0) or 0.0)
        if scale <= 0.0:
            raise TensorPreprocessError(
                "CPU_PREPROCESS_INVALID_SCALE",
                f"model preprocess scale must be positive, got {scale}",
            )
        offsets = _channel_values(model_preprocess, "offsets", np=np, dtype=dtype)
        mean = _channel_values(model_preprocess, "mean", np=np, dtype=dtype)
        std = _channel_values(model_preprocess, "std", np=np, dtype=dtype)
        if offsets is not None:
            tensor = tensor - offsets
        tensor = tensor * dtype(scale)
        if mean is not None:
            tensor = tensor - mean
        if std is not None:
            if bool(np.any(std == 0)):
                raise TensorPreprocessError(
                    "CPU_PREPROCESS_INVALID_STD",
                    "model preprocess std values must be non-zero",
                )
            tensor = tensor / std
        tensor = np.ascontiguousarray(tensor, dtype=dtype)
    layout_done_ns = time.monotonic_ns()
    timings = {
        "colorspace_ms": _elapsed_ms(convert_start_ns, convert_done_ns),
        "resize_ms": _elapsed_ms(resize_start_ns, resize_done_ns),
        "layout_ms": _elapsed_ms(layout_start_ns, layout_done_ns),
        "total_ms": _elapsed_ms(total_start_ns, layout_done_ns),
    }
    return TensorPreprocessResult(
        tensor=tensor,
        backend="cpu",
        input_mode=prepared.mode,
        resource_kind=prepared.resource_kind,
        resource_memory=prepared.resource_memory or "cpu",
        location="host",
        zero_copy=False,
        reason="cpu_host_bridge",
        timings=timings,
        metadata={
            **resize_metadata,
            "model_color_format": color_format,
            "model_scale": float(
                getattr(model_preprocess, "scale", 1.0 / 255.0) or 0.0
            ),
        },
    )


def raise_gpu_resource_preprocess_not_implemented(prepared: PreparedTensorInput) -> None:
    resource = prepared.resource_memory or "unknown"
    kind = prepared.resource_kind or "unknown"
    raise TensorPreprocessError(
        GPU_RESOURCE_PREPROCESS_NOT_IMPLEMENTED,
        f"{resource} tensor input is not implemented for {kind}; provide the "
        "NovaSight GPU-side TensorRT preprocess path",
    )


def _host_frame_to_rgb(prepared: PreparedTensorInput, *, np: Any) -> Any:
    fmt = str(prepared.pixel_format or prepared.resource_pixel_format or "").upper()
    raw = np.asarray(prepared.buffer)
    width = int(prepared.width)
    height = int(prepared.height)
    if fmt in {"BGRX", "BGRA"}:
        image = raw.reshape((height, width, 4))
        return np.ascontiguousarray(image[:, :, [2, 1, 0]])
    if fmt == "RGBA":
        image = raw.reshape((height, width, 4))
        return np.ascontiguousarray(image[:, :, :3])
    if fmt == "RGB":
        return np.ascontiguousarray(raw.reshape((height, width, 3)))
    if fmt == "BGR":
        image = raw.reshape((height, width, 3))
        return np.ascontiguousarray(image[:, :, [2, 1, 0]])
    if fmt == "NV12":
        return _nv12_to_rgb(raw, width=width, height=height, np=np)
    if fmt == "I420":
        return _i420_to_rgb(raw, width=width, height=height, np=np)
    raise TensorPreprocessError(
        "CPU_PREPROCESS_UNSUPPORTED_FORMAT",
        f"unsupported CPU preprocess input format: {fmt or '<missing>'}",
    )


def _nv12_to_rgb(raw: Any, *, width: int, height: int, np: Any) -> Any:
    flat = np.asarray(raw, dtype=np.uint8).reshape(-1)
    expected = width * height * 3 // 2
    if flat.size < expected:
        raise TensorPreprocessError(
            "CPU_PREPROCESS_BUFFER_TOO_SMALL",
            f"NV12 frame buffer too small: got {flat.size}, expected {expected}",
        )
    y_size = width * height
    y = flat[:y_size].reshape((height, width)).astype(np.int32)
    uv = flat[y_size:expected].reshape((height // 2, width // 2, 2)).astype(np.int32)
    u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)[:height, :width]
    v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)[:height, :width]
    return _yuv_to_rgb(y, u, v, np=np)


def _i420_to_rgb(raw: Any, *, width: int, height: int, np: Any) -> Any:
    flat = np.asarray(raw, dtype=np.uint8).reshape(-1)
    expected = width * height * 3 // 2
    if flat.size < expected:
        raise TensorPreprocessError(
            "CPU_PREPROCESS_BUFFER_TOO_SMALL",
            f"I420 frame buffer too small: got {flat.size}, expected {expected}",
        )
    y_size = width * height
    chroma_size = (width // 2) * (height // 2)
    y = flat[:y_size].reshape((height, width)).astype(np.int32)
    u_plane = flat[y_size : y_size + chroma_size].reshape((height // 2, width // 2)).astype(np.int32)
    v_plane = flat[y_size + chroma_size : y_size + chroma_size * 2].reshape((height // 2, width // 2)).astype(np.int32)
    u = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)[:height, :width]
    v = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)[:height, :width]
    return _yuv_to_rgb(y, u, v, np=np)


def _yuv_to_rgb(y: Any, u: Any, v: Any, *, np: Any) -> Any:
    c = np.maximum(y - 16, 0)
    d = u - 128
    e = v - 128
    r = (298 * c + 409 * e + 128) >> 8
    g = (298 * c - 100 * d - 208 * e + 128) >> 8
    b = (298 * c + 516 * d + 128) >> 8
    return np.ascontiguousarray(
        np.stack(
            [
                np.clip(r, 0, 255),
                np.clip(g, 0, 255),
                np.clip(b, 0, 255),
            ],
            axis=2,
        ).astype(np.uint8)
    )


def _resize_rgb(image: Any, *, width: int, height: int, np: Any) -> Any:
    try:
        import cv2

        resized = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(resized)
    except Exception:
        src_h, src_w = image.shape[:2]
        y_idx = np.linspace(0, src_h - 1, int(height)).astype(np.int64)
        x_idx = np.linspace(0, src_w - 1, int(width)).astype(np.int64)
        return np.ascontiguousarray(image[y_idx][:, x_idx])


def _resize_or_letterbox_rgb(
    image: Any,
    *,
    width: int,
    height: int,
    np: Any,
    pad_value: int = 114,
) -> tuple[Any, dict[str, Any]]:
    src_h, src_w = image.shape[:2]
    target_w = int(width)
    target_h = int(height)
    if src_w == target_w and src_h == target_h:
        return np.ascontiguousarray(image), {
            "resize_mode": "none",
            "model_content_width": target_w,
            "model_content_height": target_h,
            "pad_x": 0,
            "pad_y": 0,
            "pad_value": 0,
        }

    src_ratio = float(src_w) / max(1.0, float(src_h))
    target_ratio = float(target_w) / max(1.0, float(target_h))
    if abs(src_ratio - target_ratio) <= 1e-6:
        return _resize_rgb(image, width=target_w, height=target_h, np=np), {
            "resize_mode": "resize",
            "model_content_width": target_w,
            "model_content_height": target_h,
            "pad_x": 0,
            "pad_y": 0,
            "pad_value": 0,
        }

    scale = min(float(target_w) / max(1.0, float(src_w)), float(target_h) / max(1.0, float(src_h)))
    content_w = max(1, min(target_w, int(round(src_w * scale))))
    content_h = max(1, min(target_h, int(round(src_h * scale))))
    resized = _resize_rgb(image, width=content_w, height=content_h, np=np)
    pad_x = max(0, (target_w - content_w) // 2)
    pad_y = max(0, (target_h - content_h) // 2)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[pad_y : pad_y + content_h, pad_x : pad_x + content_w, :] = resized
    return np.ascontiguousarray(canvas), {
        "resize_mode": "letterbox",
        "model_content_width": content_w,
        "model_content_height": content_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "pad_value": pad_value,
    }


def _resize_for_profile(
    image: Any,
    *,
    width: int,
    height: int,
    np: Any,
    model_preprocess: Any | None,
) -> tuple[Any, dict[str, Any]]:
    mode = str(getattr(model_preprocess, "resize_mode", "") or "auto").lower()
    if mode not in {"auto", "direct", "letterbox"}:
        raise TensorPreprocessError(
            "CPU_PREPROCESS_UNSUPPORTED_RESIZE_MODE",
            f"unsupported model resize mode: {mode}",
        )
    src_h, src_w = image.shape[:2]
    target_w = int(width)
    target_h = int(height)
    if src_w == target_w and src_h == target_h:
        return np.ascontiguousarray(image), {
            "resize_mode": "none",
            "model_content_width": target_w,
            "model_content_height": target_h,
            "pad_x": 0,
            "pad_y": 0,
            "pad_value": 0,
        }
    if mode == "direct":
        return _resize_rgb(image, width=target_w, height=target_h, np=np), {
            "resize_mode": "resize",
            "model_content_width": target_w,
            "model_content_height": target_h,
            "pad_x": 0,
            "pad_y": 0,
            "pad_value": 0,
        }
    padding_value = int(
        max(0, min(255, round(float(getattr(model_preprocess, "padding_value", 114.0)))))
    )
    return _resize_or_letterbox_rgb(
        image,
        width=target_w,
        height=target_h,
        np=np,
        pad_value=padding_value,
    )


def _channel_values(
    model_preprocess: Any | None,
    name: str,
    *,
    np: Any,
    dtype: Any,
) -> Any | None:
    values = tuple(float(value) for value in (getattr(model_preprocess, name, ()) or ()))
    if not values:
        return None
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise TensorPreprocessError(
            "CPU_PREPROCESS_INVALID_CHANNEL_VALUES",
            f"model preprocess {name} must contain one or three values",
        )
    return np.asarray(values, dtype=dtype).reshape((1, 3, 1, 1))


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return max(0.0, (int(end_ns) - int(start_ns)) / 1e6)


def _validate_device_preprocess_result(
    result: TensorPreprocessResult,
    shape: TensorInputShape,
) -> None:
    if result.location != "device" or not isinstance(result.tensor, DeviceTensor):
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_INVALID",
            "GPU resource preprocessor must return a device TensorPreprocessResult",
        )
    if result.zero_copy is not True:
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_INVALID",
            "GPU resource preprocessor must explicitly return zero_copy=True",
        )
    owner = getattr(result.tensor, "owner", None)
    release = getattr(owner, "release", None)
    if not callable(release):
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_INVALID",
            "GPU resource preprocessor must return a DeviceTensor owner with a callable release()",
        )
    if int(result.tensor.device_ptr) <= 0:
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_INVALID",
            f"GPU preprocessor returned invalid device_ptr={result.tensor.device_ptr}",
        )
    if int(result.tensor.nbytes) <= 0:
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_INVALID",
            f"GPU preprocessor returned invalid nbytes={result.tensor.nbytes}",
        )
    expected_shape = (shape.batch, shape.channels, shape.height, shape.width)
    if tuple(int(item) for item in result.tensor.shape) != expected_shape:
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_SHAPE_MISMATCH",
            f"GPU preprocessor returned shape {result.tensor.shape}, expected {expected_shape}",
        )
    expected_nbytes = shape.nbytes
    if int(result.tensor.nbytes) < expected_nbytes:
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_INVALID",
            f"GPU preprocessor returned too few bytes: got {result.tensor.nbytes}, "
            f"expected at least {expected_nbytes} for {shape.dtype}",
        )
    try:
        actual_dtype = normalize_tensor_dtype(result.tensor.dtype)
        expected_dtype = normalize_tensor_dtype(shape.dtype)
    except ValueError as exc:
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_DTYPE_MISMATCH",
            str(exc),
        ) from exc
    if actual_dtype != expected_dtype:
        raise TensorPreprocessError(
            "GPU_RESOURCE_PREPROCESS_DTYPE_MISMATCH",
            f"GPU preprocessor returned dtype {result.tensor.dtype}, expected {expected_dtype}",
        )
