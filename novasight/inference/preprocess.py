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

    def debug_payload(self) -> dict[str, Any]:
        return {
            "preprocess_backend": self.backend,
            "preprocess_location": self.location,
            "preprocess_zero_copy": self.zero_copy,
            "preprocess_reason": self.reason,
        }


class GpuResourcePreprocessor(Protocol):
    def prepare(
        self,
        prepared: PreparedTensorInput,
        shape: TensorInputShape,
    ) -> TensorPreprocessResult: ...


def prepare_tensor(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    gpu_preprocessor: GpuResourcePreprocessor | None = None,
) -> TensorPreprocessResult:
    import numpy as np

    if prepared.mode == "gpu_buffer":
        if gpu_preprocessor is None:
            raise_gpu_resource_preprocess_not_implemented(prepared)
        result = gpu_preprocessor.prepare(prepared, shape)
        _validate_device_preprocess_result(result, shape)
        return result
    return prepare_host_tensor(prepared, shape, np=np)


def prepare_host_tensor(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    np: Any | None = None,
) -> TensorPreprocessResult:
    if prepared.mode == "gpu_buffer":
        raise_gpu_resource_preprocess_not_implemented(prepared)
    if np is None:
        import numpy as np_module

        np = np_module

    tensor = _prepare_cpu_numpy_tensor(prepared, shape, np=np)
    return TensorPreprocessResult(
        tensor=tensor,
        backend="cpu_numpy",
        input_mode=prepared.mode,
        resource_kind=prepared.resource_kind,
        resource_memory=prepared.resource_memory,
        location="host",
        zero_copy=False,
    )


def raise_gpu_resource_preprocess_not_implemented(prepared: PreparedTensorInput) -> None:
    resource = prepared.resource_memory or "unknown"
    kind = prepared.resource_kind or "unknown"
    raise TensorPreprocessError(
        GPU_RESOURCE_PREPROCESS_NOT_IMPLEMENTED,
        f"{resource} tensor input is not implemented for {kind}; use CPU image "
        "fallback or provide a real GPU-side TensorRT/ONNX preprocess path",
    )


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


def _prepare_cpu_numpy_tensor(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    np: Any,
) -> Any:
    image = prepared.buffer
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            if image.size != (shape.width, shape.height):
                image = image.resize((shape.width, shape.height))
            array = np.asarray(image, dtype=np.float32) / 255.0
            return array.transpose(2, 0, 1)[None, ...]
    except Exception:
        pass
    if hasattr(image, "shape"):
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim == 3 and array.shape[2] >= 3:
            array = np.ascontiguousarray(array[:, :, :3])
            if prepared.pixel_format in {"BGR", "BGR3"}:
                array = array[:, :, ::-1]
            if array.shape[1] != shape.width or array.shape[0] != shape.height:
                array = _resize_numpy_image(array, width=shape.width, height=shape.height, np=np)
            array = array.astype(np.float32) / 255.0
            return array.transpose(2, 0, 1)[None, ...]
    return np.zeros((shape.batch, shape.channels, shape.height, shape.width), dtype=np.float32)


def _resize_numpy_image(image: Any, *, width: int, height: int, np: Any) -> Any:
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((width, height)))
    except Exception:
        y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
        x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
        return image[y_idx][:, x_idx]
