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

    def debug_payload(self) -> dict[str, Any]:
        payload = {
            "preprocess_backend": self.backend,
            "preprocess_location": self.location,
            "preprocess_zero_copy": self.zero_copy,
            "preprocess_reason": self.reason,
        }
        if self.timings:
            payload["preprocess_native_timings"] = dict(self.timings)
        return payload


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
    if prepared.mode != "gpu_buffer" or gpu_preprocessor is None:
        raise_gpu_resource_preprocess_not_implemented(prepared)
    result = gpu_preprocessor.prepare(prepared, shape)
    _validate_device_preprocess_result(result, shape)
    return result


def raise_gpu_resource_preprocess_not_implemented(prepared: PreparedTensorInput) -> None:
    resource = prepared.resource_memory or "unknown"
    kind = prepared.resource_kind or "unknown"
    raise TensorPreprocessError(
        GPU_RESOURCE_PREPROCESS_NOT_IMPLEMENTED,
        f"{resource} tensor input is not implemented for {kind}; provide the "
        "NovaSight GPU-side TensorRT preprocess path",
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
