from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Mapping

from novasight.config.runtime import RuntimeConfig

from .input import PreparedTensorInput, TensorInputShape, normalize_tensor_dtype
from .preprocess import DeviceTensor, TensorPreprocessError, TensorPreprocessResult


JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE = "JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE"
JETSON_GPU_RESOURCE_BRIDGE_INVALID = "JETSON_GPU_RESOURCE_BRIDGE_INVALID"
JETSON_GPU_RESOURCE_BRIDGE_FAILED = "JETSON_GPU_RESOURCE_BRIDGE_FAILED"
DEFAULT_JETSON_GPU_RESOURCE_BRIDGE_MODULE = "novasight_jetson_preprocess"
JETSON_GPU_RESOURCE_BRIDGE_ENV = "NOVASIGHT_JETSON_PREPROCESSOR"
JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION = 1
JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_PAYLOAD_FIELDS = (
    "resource_handle",
    "frame_id",
    "capture_ts_ns",
    "resource_kind",
    "resource_memory",
    "resource_source",
    "dmabuf_fd",
    "gst_buffer_ptr",
    "resource_metadata",
    "resource_width",
    "resource_height",
    "resource_pixel_format",
    "width",
    "height",
    "pixel_format",
    "source_width",
    "source_height",
    "roi_offset_x",
    "roi_offset_y",
    "needs_resize",
    "model_shape",
    "nchw",
    "dtype",
)
JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_RESULT_FIELDS = (
    "device_ptr",
    "nbytes",
    "zero_copy",
    "memory_space",
)
JETSON_GPU_RESOURCE_BRIDGE_OPTIONAL_RESULT_FIELDS = (
    "shape",
    "dtype",
    "stream",
    "owner",
    "backend",
    "zero_copy",
    "reason",
)
JETSON_GPU_RESOURCE_BRIDGE_DEVICE_MEMORY_FIELDS = (
    "memory_space",
    "memory_location",
    "location",
    "pointer_location",
    "pointer_type",
)
JETSON_GPU_RESOURCE_BRIDGE_DEVICE_MEMORY_VALUES = {
    "device",
    "cuda",
    "cuda_device",
    "gpu",
    "gpu_device",
}


@dataclass
class JetsonGpuResourcePreprocessor:
    """Jetson NVMM/DMABUF resource bridge boundary.

    The Python side defines a stable native bridge ABI and remains fail-closed
    when the native module is absent. A production Jetson install can provide a
    module exposing `prepare_tensor(payload)` or `prepare_nvmm_tensor(payload)`
    and return a real device pointer for TensorRT.
    """

    enabled: bool = True
    module_name: str = DEFAULT_JETSON_GPU_RESOURCE_BRIDGE_MODULE
    _bridge: Any | None = field(default=None, init=False, repr=False)
    _load_error: str = field(default="", init=False, repr=False)
    _load_reason: str = field(
        default=JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE,
        init=False,
        repr=False,
    )
    _abi_version: int | None = field(default=None, init=False, repr=False)
    _abi_compatible: bool = field(default=False, init=False, repr=False)
    _capabilities: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _native_ready: bool = field(default=False, init=False, repr=False)
    _native_status: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _model_preprocess: Any | None = field(default=None, init=False, repr=False)

    @property
    def backend(self) -> str:
        return "jetson_nvmm_cuda"

    def status(self) -> dict[str, Any]:
        bridge = self._load_bridge()
        return {
            "selected": self.backend,
            "enabled": self.enabled,
            "available": bridge is not None,
            "module": self.module_name,
            "env": JETSON_GPU_RESOURCE_BRIDGE_ENV,
            "required_abi_version": JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
            "abi_version": self._abi_version,
            "abi_compatible": self._abi_compatible,
            "native_ready": self._native_ready,
            "native_status": dict(self._native_status),
            "capabilities": dict(self._capabilities),
            "contract": jetson_gpu_resource_bridge_contract(),
            "reason": "" if bridge is not None else self._load_reason,
            "detail": "" if bridge is not None else self._load_error,
            "model_preprocess": _model_preprocess_payload(self._model_preprocess),
        }

    def configure_model_preprocess(self, model_preprocess: Any | None) -> None:
        self._model_preprocess = model_preprocess

    def prepare(
        self,
        prepared: PreparedTensorInput,
        shape: TensorInputShape,
    ) -> TensorPreprocessResult:
        _validate_native_model_preprocess(self._model_preprocess)
        bridge = self._load_bridge()
        if bridge is None:
            self._raise_unavailable(prepared, shape)
        payload = _native_bridge_payload(
            prepared,
            shape,
            model_preprocess=self._model_preprocess,
        )
        _validate_native_bridge_payload(
            self._capabilities,
            payload,
            module_name=self.module_name,
        )
        try:
            native_result = _call_native_bridge(bridge, payload)
        except TensorPreprocessError:
            raise
        except Exception as exc:
            raise TensorPreprocessError(
                JETSON_GPU_RESOURCE_BRIDGE_FAILED,
                f"Jetson GPU resource bridge failed while preparing tensor: {exc}",
            ) from exc
        return _tensor_result_from_native(
            native_result,
            prepared=prepared,
            shape=shape,
            default_backend=f"{self.backend}:{self.module_name}",
        )

    def _load_bridge(self) -> Any | None:
        if not self.enabled:
            self._reset_bridge_status(
                JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE,
                "Jetson GPU resource preprocessor is disabled",
            )
            return None
        if self._bridge is not None:
            return self._bridge
        module_name = os.environ.get(JETSON_GPU_RESOURCE_BRIDGE_ENV, self.module_name).strip()
        if not module_name:
            self._reset_bridge_status(
                JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE,
                f"{JETSON_GPU_RESOURCE_BRIDGE_ENV} is empty",
            )
            return None
        self.module_name = module_name
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            self._reset_bridge_status(
                JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE,
                f"{module_name}: {exc}",
            )
            return None
        self._capabilities = _native_bridge_capabilities(module)
        self._abi_version = _native_bridge_abi_version(module)
        self._abi_compatible = self._abi_version == JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
        self._native_ready, self._native_status = _native_bridge_readiness(module)
        if not self._abi_compatible:
            expected = JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
            actual = self._abi_version if self._abi_version is not None else "missing"
            self._bridge = None
            self._load_reason = JETSON_GPU_RESOURCE_BRIDGE_INVALID
            self._load_error = (
                f"{module_name} ABI_VERSION={actual} is incompatible with "
                f"required ABI_VERSION={expected}"
            )
            return None
        if not self._native_ready:
            self._bridge = None
            self._load_reason = JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE
            detail = _native_status_detail(self._native_status)
            self._load_error = detail or f"{module_name} native bridge is not ready"
            return None
        bridge = _select_native_bridge_callable(module)
        if bridge is None:
            self._bridge = None
            self._load_reason = JETSON_GPU_RESOURCE_BRIDGE_INVALID
            self._load_error = (
                f"{module_name} must expose prepare_tensor(payload) or "
                "prepare_nvmm_tensor(payload)"
            )
            return None
        self._bridge = bridge
        self._load_reason = ""
        self._load_error = ""
        return bridge

    def _reset_bridge_status(self, reason: str, detail: str) -> None:
        self._bridge = None
        self._load_reason = reason
        self._load_error = detail
        self._abi_version = None
        self._abi_compatible = False
        self._native_ready = False
        self._native_status = {}
        self._capabilities = {}

    def _raise_unavailable(
        self,
        prepared: PreparedTensorInput,
        shape: TensorInputShape,
    ) -> None:
        resource = prepared.resource_memory or "unknown"
        kind = prepared.resource_kind or "unknown"
        detail = f"; bridge detail: {self._load_error}" if self._load_error else ""
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE,
            "Jetson GPU resource bridge is not installed: "
            f"cannot convert {resource}/{kind} {prepared.width}x{prepared.height} "
            f"{prepared.pixel_format} into NCHW tensor {shape}{detail}",
        )


def create_gpu_resource_preprocessor(config: RuntimeConfig) -> JetsonGpuResourcePreprocessor | None:
    """Create the NovaSight GPU-side TensorRT preprocess boundary when needed."""
    inference = getattr(config, "inference", None)
    capture = getattr(config, "capture", None)
    if not bool(getattr(inference, "enabled", True)):
        return None
    backend = str(getattr(inference, "backend", "") or "").lower()
    memory = str(getattr(capture, "memory", "") or "").lower()
    if backend != "nvmm_latest" or memory != "nvmm":
        return None
    return JetsonGpuResourcePreprocessor()


def jetson_gpu_resource_bridge_contract() -> dict[str, Any]:
    return {
        "abi_version": JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        "callable": "prepare_tensor(payload) or prepare_nvmm_tensor(payload)",
        "payload_required_fields": list(JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_PAYLOAD_FIELDS),
        "result_required_fields": list(JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_RESULT_FIELDS),
        "result_optional_fields": list(JETSON_GPU_RESOURCE_BRIDGE_OPTIONAL_RESULT_FIELDS),
        "result_semantics": (
            "Return an existing GPU device pointer containing a model-ready "
            "NCHW tensor. Dict results must explicitly include zero_copy=true "
            "and memory_space=device/cuda/cuda_device/gpu/gpu_device. Do not "
            "return synthetic CPU data, host pointers, or zero-filled device "
            "memory as a successful native bridge result."
        ),
    }


def _select_native_bridge_callable(module: ModuleType | Any) -> Any | None:
    for name in ("prepare_tensor", "prepare_nvmm_tensor"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    if callable(module):
        return module
    return None


def _native_bridge_abi_version(module: ModuleType | Any) -> int | None:
    for name in ("ABI_VERSION", "abi_version", "__abi_version__", "get_abi_version"):
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        try:
            if callable(value):
                value = value()
            return int(value)
        except Exception:
            return None
    return None


def _native_bridge_capabilities(module: ModuleType | Any) -> dict[str, Any]:
    for name in ("CAPABILITIES", "capabilities", "get_capabilities"):
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        try:
            if callable(value):
                value = value()
        except Exception as exc:
            return {"error": str(exc)}
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, (list, tuple, set)):
            return {"features": list(value)}
        return {"value": str(value)}
    return {}


def _native_bridge_readiness(module: ModuleType | Any) -> tuple[bool, dict[str, Any]]:
    for name in ("status", "bridge_status", "get_status", "get_bridge_status"):
        if not hasattr(module, name):
            continue
        candidate = getattr(module, name)
        if not callable(candidate):
            continue
        try:
            value = candidate()
        except Exception as exc:
            return False, {"available": False, "reason": "status_exception", "detail": str(exc)}
        if isinstance(value, Mapping):
            status = dict(value)
            if "available" not in status and "ready" not in status:
                status.update(
                    {
                        "available": False,
                        "reason": "native_readiness_status_required",
                        "detail": (
                            "status() must explicitly report available or ready; "
                            "implicit readiness is not accepted for NVMM inference"
                        ),
                    }
                )
                return False, status
            ready = bool(status.get("ready", status.get("available", False)))
            if not ready:
                return False, status
            contract_error = _native_status_output_contract_error(status)
            if contract_error:
                status.update(
                    {
                        "available": False,
                        "ready": False,
                        "reason": "native_output_contract_invalid",
                        "detail": contract_error,
                    }
                )
                return False, status
            return True, status
        ready = bool(value)
        if ready:
            return (
                False,
                {
                    "available": False,
                    "ready": False,
                    "reason": "native_readiness_status_required",
                    "detail": (
                        "status() must return a mapping with zero_copy and "
                        "memory_space when reporting ready"
                    ),
                },
            )
        return False, {"available": False, "ready": False}
    return (
        False,
        {
            "available": False,
            "ready": False,
            "reason": "native_readiness_status_required",
            "detail": (
                "Jetson bridge modules must expose status(), bridge_status(), "
                "get_status(), or get_bridge_status() before NVMM inference can start"
            ),
        },
    )


def _native_status_output_contract_error(status: Mapping[str, Any]) -> str:
    backend = _normalize_capability_value(status.get("backend", ""))
    if backend and ("reference" in backend or "scaffold" in backend):
        return f"backend={status.get('backend')!r} is not a production Jetson backend"
    if _status_zero_copy_value(status) is not True:
        return "ready native bridge status must include zero_copy=true"
    memory_space = _status_device_memory_space(status)
    if memory_space not in JETSON_GPU_RESOURCE_BRIDGE_DEVICE_MEMORY_VALUES:
        accepted = ", ".join(sorted(JETSON_GPU_RESOURCE_BRIDGE_DEVICE_MEMORY_VALUES))
        return (
            "ready native bridge status must include device memory_space; "
            f"accepted values: {accepted}"
        )
    return ""


def _status_zero_copy_value(status: Mapping[str, Any]) -> bool | None:
    for container in _status_contract_containers(status):
        if "zero_copy" in container:
            return container.get("zero_copy") is True
    return None


def _status_device_memory_space(status: Mapping[str, Any]) -> str:
    for container in _status_contract_containers(status):
        for key in JETSON_GPU_RESOURCE_BRIDGE_DEVICE_MEMORY_FIELDS:
            normalized = _normalize_capability_value(container.get(key))
            if normalized:
                return normalized
    return ""


def _status_contract_containers(status: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = [status]
    for key in ("capabilities", "output", "result", "tensor", "device_tensor"):
        value = status.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    return containers


def _native_status_detail(status: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("reason", "detail", "message", "backend", "module"):
        value = status.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def _validate_native_bridge_payload(
    capabilities: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    module_name: str,
) -> None:
    _validate_required_payload_contract(payload, module_name=module_name)
    for payload_key, aliases in (
        (
            "resource_memory",
            ("memory", "memories", "resource_memory", "resource_memories", "input_memory", "input_memories"),
        ),
        (
            "resource_kind",
            ("kind", "kinds", "resource_kind", "resource_kinds", "input_kind", "input_kinds"),
        ),
        (
            "resource_source",
            ("source", "sources", "resource_source", "resource_sources", "input_source", "input_sources"),
        ),
        (
            "pixel_format",
            ("format", "formats", "pixel_format", "pixel_formats", "input_format", "input_formats"),
        ),
        (
            "dtype",
            ("dtype", "dtypes", "tensor_dtype", "tensor_dtypes", "output_dtype", "output_dtypes"),
        ),
    ):
        supported = _capability_values(capabilities, aliases)
        if supported is None:
            continue
        actual = _normalize_capability_value(payload.get(payload_key))
        if not actual:
            continue
        if _capability_allows(supported, actual):
            continue
        supported_text = ", ".join(sorted(supported))
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} does not support "
            f"{payload_key}={payload.get(payload_key)!r}; supported: {supported_text}",
        )


def _validate_required_payload_contract(
    payload: Mapping[str, Any],
    *,
    module_name: str,
) -> None:
    missing = [
        field
        for field in JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_PAYLOAD_FIELDS
        if field not in payload
    ]
    if missing:
        fields = ", ".join(missing)
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} payload missing required field(s): {fields}",
        )

    _require_payload_positive_int(payload, "frame_id", module_name)
    _require_payload_positive_int(payload, "capture_ts_ns", module_name)
    resource_kind = _require_payload_nonempty_text(payload, "resource_kind", module_name)
    _require_payload_nonempty_text(payload, "resource_memory", module_name)
    resource_source = _require_payload_nonempty_text(payload, "resource_source", module_name)
    if _normalize_capability_value(resource_source) != "appsink":
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires resource_source='appsink'; "
            f"got {resource_source!r}",
        )
    if _normalize_capability_value(resource_kind) == "gstreamer_sample" and payload.get("resource_handle") is None:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires a live resource_handle "
            "for resource_kind='gstreamer_sample' so the Gst.Sample/GstBuffer "
            "outlives native preprocess",
        )
    if not isinstance(payload.get("resource_metadata"), Mapping):
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires resource_metadata to be an object",
        )
    dmabuf_fd = _optional_nonnegative_int(payload.get("dmabuf_fd"))
    gst_buffer_ptr = _optional_positive_int(payload.get("gst_buffer_ptr"))
    if dmabuf_fd is None and gst_buffer_ptr is None:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires dmabuf_fd or "
            "gst_buffer_ptr. The capture pipeline delivered NVMM without a "
            "DMABUF fd and did not expose the GstBuffer pointer fallback.",
        )
    resource_width = _require_payload_positive_int(payload, "resource_width", module_name)
    resource_height = _require_payload_positive_int(payload, "resource_height", module_name)
    resource_pixel_format = _require_payload_nonempty_text(
        payload,
        "resource_pixel_format",
        module_name,
    )
    _require_payload_positive_int(payload, "width", module_name)
    _require_payload_positive_int(payload, "height", module_name)
    _require_payload_nonempty_text(payload, "pixel_format", module_name)
    _require_payload_positive_int(payload, "source_width", module_name)
    _require_payload_positive_int(payload, "source_height", module_name)
    _require_payload_nonnegative_int(payload, "roi_offset_x", module_name)
    _require_payload_nonnegative_int(payload, "roi_offset_y", module_name)
    if not isinstance(payload.get("needs_resize"), bool):
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires needs_resize to be boolean",
        )
    if resource_width != int(payload.get("width")):
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires resource_width to match width; "
            f"got resource_width={resource_width}, width={payload.get('width')!r}",
        )
    if resource_height != int(payload.get("height")):
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires resource_height to match height; "
            f"got resource_height={resource_height}, height={payload.get('height')!r}",
        )
    if _normalize_capability_value(resource_pixel_format) != _normalize_capability_value(
        payload.get("pixel_format")
    ):
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires resource_pixel_format to match pixel_format; "
            f"got resource_pixel_format={resource_pixel_format!r}, pixel_format={payload.get('pixel_format')!r}",
        )
    model_shape = payload.get("model_shape")
    if not isinstance(model_shape, Mapping):
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires model_shape to be an object",
        )
    for key in ("batch", "channels", "height", "width"):
        _require_mapping_positive_int(model_shape, key, module_name, container="model_shape")
    nchw = payload.get("nchw")
    if not isinstance(nchw, (list, tuple)) or len(nchw) != 4:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires nchw to contain four dimensions",
        )
    for index, value in enumerate(nchw):
        _require_positive_int_value(
            value,
            f"nchw[{index}]",
            module_name,
        )
    _require_payload_nonempty_text(payload, "dtype", module_name)


def _require_payload_positive_int(
    payload: Mapping[str, Any],
    key: str,
    module_name: str,
) -> int:
    return _require_positive_int_value(payload.get(key), key, module_name)


def _require_payload_nonnegative_int(
    payload: Mapping[str, Any],
    key: str,
    module_name: str,
) -> int:
    return _require_nonnegative_int_value(payload.get(key), key, module_name)


def _require_mapping_positive_int(
    payload: Mapping[str, Any],
    key: str,
    module_name: str,
    *,
    container: str,
) -> int:
    return _require_positive_int_value(
        payload.get(key),
        f"{container}.{key}",
        module_name,
    )


def _require_payload_nonempty_text(
    payload: Mapping[str, Any],
    key: str,
    module_name: str,
) -> str:
    value = payload.get(key)
    text = str(value or "").strip()
    if not text:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires non-empty {key}",
        )
    return text


def _require_positive_int_value(value: Any, key: str, module_name: str) -> int:
    try:
        result = int(value)
    except Exception as exc:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires {key} to be a positive integer",
        ) from exc
    if result <= 0:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires {key} to be a positive integer; got {value!r}",
        )
    return result


def _require_nonnegative_int_value(value: Any, key: str, module_name: str) -> int:
    if value is None:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires {key} "
            "to be a non-negative integer; got None. The capture pipeline "
            "did not deliver a dmabuf file descriptor (appsink received a "
            "non-DMABUF buffer, so NvBufSurface is unreachable).",
        )
    if not isinstance(value, (int, float)):
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires {key} "
            f"to be a non-negative integer; got {value!r} (type={type(value).__name__}).",
        )
    try:
        result = int(value)
    except Exception as exc:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires {key} "
            f"to be a non-negative integer; cannot parse {value!r} as int.",
        ) from exc
    if result < 0:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge {module_name} requires {key} "
            f"to be a non-negative integer; got {value!r}",
        )
    return result


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except Exception:
        return None
    return result if result >= 0 else None


def _optional_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except Exception:
        return None
    return result if result > 0 else None


def _capability_values(
    capabilities: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> set[str] | None:
    for key in aliases:
        if key not in capabilities:
            continue
        return _normalize_capability_values(capabilities[key])
    return None


def _normalize_capability_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        return {_normalize_capability_value(item) for item in parts if item.strip()}
    if isinstance(value, Mapping):
        return {
            _normalize_capability_value(key)
            for key, enabled in value.items()
            if bool(enabled)
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return {
            _normalize_capability_value(item)
            for item in value
            if _normalize_capability_value(item)
        }
    normalized = _normalize_capability_value(value)
    return {normalized} if normalized else set()


def _normalize_capability_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_")


def _capability_allows(supported: set[str], actual: str) -> bool:
    if not supported:
        return False
    return actual in supported or "*" in supported or "any" in supported or "all" in supported


def _call_native_bridge(bridge: Any, payload: dict[str, Any]) -> Any:
    return bridge(payload)


def _native_bridge_payload(
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    *,
    model_preprocess: Any | None = None,
) -> dict[str, Any]:
    resource_metadata = dict(prepared.resource_metadata)
    return {
        "resource_handle": prepared.buffer,
        "frame_id": int(prepared.frame_id),
        "capture_ts_ns": int(prepared.capture_ts_ns),
        "resource_kind": prepared.resource_kind,
        "resource_memory": prepared.resource_memory,
        "resource_source": prepared.resource_source,
        "dmabuf_fd": prepared.dmabuf_fd,
        "gst_buffer_ptr": _optional_positive_int(resource_metadata.get("gst_buffer_ptr")),
        "resource_metadata": resource_metadata,
        "resource_width": int(prepared.resource_width),
        "resource_height": int(prepared.resource_height),
        "resource_pixel_format": prepared.resource_pixel_format,
        "width": int(prepared.width),
        "height": int(prepared.height),
        "pixel_format": prepared.pixel_format,
        "source_width": int(prepared.source_width),
        "source_height": int(prepared.source_height),
        "roi_offset_x": int(prepared.offset_x),
        "roi_offset_y": int(prepared.offset_y),
        "needs_resize": bool(prepared.needs_resize),
        "model_shape": {
            "batch": int(shape.batch),
            "channels": int(shape.channels),
            "height": int(shape.height),
            "width": int(shape.width),
        },
        "nchw": (int(shape.batch), int(shape.channels), int(shape.height), int(shape.width)),
        "dtype": normalize_tensor_dtype(shape.dtype),
        "model_preprocess": _model_preprocess_payload(model_preprocess),
    }


def _model_preprocess_payload(model_preprocess: Any | None) -> dict[str, Any]:
    if model_preprocess is None:
        return {
            "color_format": "RGB",
            "scale": 1.0 / 255.0,
            "offsets": [],
            "mean": [],
            "std": [],
            "resize_mode": "direct",
            "symmetric_padding": False,
            "padding_value": 0.0,
        }
    return {
        "color_format": str(getattr(model_preprocess, "color_format", "") or "").upper(),
        "scale": float(getattr(model_preprocess, "scale", 0.0) or 0.0),
        "offsets": [float(value) for value in getattr(model_preprocess, "offsets", ())],
        "mean": [float(value) for value in getattr(model_preprocess, "mean", ())],
        "std": [float(value) for value in getattr(model_preprocess, "std", ())],
        "resize_mode": str(getattr(model_preprocess, "resize_mode", "") or "").lower(),
        "symmetric_padding": bool(
            getattr(model_preprocess, "symmetric_padding", False)
        ),
        "padding_value": float(getattr(model_preprocess, "padding_value", 0.0) or 0.0),
    }


def _validate_native_model_preprocess(model_preprocess: Any | None) -> None:
    payload = _model_preprocess_payload(model_preprocess)
    if payload["color_format"] != "RGB":
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_FAILED,
            "Jetson native preprocess currently only supports RGB model input",
        )
    if abs(float(payload["scale"]) - (1.0 / 255.0)) > 1e-12:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_FAILED,
            "Jetson native preprocess currently only supports scale=1/255",
        )
    if payload["offsets"] or payload["mean"] or payload["std"]:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_FAILED,
            "Jetson native preprocess does not support offsets/mean/std",
        )
    if payload["resize_mode"] != "direct":
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_FAILED,
            "Jetson native preprocess currently only supports direct resize",
        )


def _tensor_result_from_native(
    native_result: Any,
    *,
    prepared: PreparedTensorInput,
    shape: TensorInputShape,
    default_backend: str,
) -> TensorPreprocessResult:
    if isinstance(native_result, TensorPreprocessResult):
        return native_result
    if isinstance(native_result, DeviceTensor):
        tensor = native_result
        backend = default_backend
        zero_copy = True
        reason = ""
    else:
        tensor = DeviceTensor(
            device_ptr=_required_positive_int(native_result, "device_ptr"),
            nbytes=_required_positive_int(native_result, "nbytes"),
            shape=_shape_from_native(native_result, shape),
            dtype=str(_optional_value(native_result, "dtype", "float32")),
            stream=_optional_value(native_result, "stream", None),
            owner=_optional_value(native_result, "owner", native_result),
        )
        backend = str(_optional_value(native_result, "backend", default_backend))
        _require_true(native_result, "zero_copy")
        _require_device_memory_space(native_result)
        zero_copy = True
        reason = str(_optional_value(native_result, "reason", ""))
    timings = _timings_from_native(native_result)
    return TensorPreprocessResult(
        tensor=tensor,
        backend=backend,
        input_mode=prepared.mode,
        resource_kind=prepared.resource_kind,
        resource_memory=prepared.resource_memory,
        location="device",
        zero_copy=zero_copy,
        reason=reason,
        timings=timings,
    )


def _timings_from_native(native_result: Any) -> dict[str, float]:
    raw = _optional_value(native_result, "timings", None)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            "Jetson GPU resource bridge result timings must be an object",
        )
    timings: dict[str, float] = {}
    for key, value in raw.items():
        try:
            timings[str(key)] = float(value)
        except Exception as exc:
            raise TensorPreprocessError(
                JETSON_GPU_RESOURCE_BRIDGE_INVALID,
                f"Jetson GPU resource bridge timing {key!r} must be numeric",
            ) from exc
    return timings


def _shape_from_native(native_result: Any, shape: TensorInputShape) -> tuple[int, ...]:
    value = _optional_value(native_result, "shape", None)
    if value is None:
        return (int(shape.batch), int(shape.channels), int(shape.height), int(shape.width))
    try:
        result = tuple(int(item) for item in value)
    except Exception as exc:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge returned invalid shape: {value}",
        ) from exc
    return result


def _required_positive_int(native_result: Any, key: str) -> int:
    value = _optional_value(native_result, key, None)
    if value is None:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge result missing required field: {key}",
        )
    try:
        result = int(value)
    except Exception as exc:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge field {key} must be an integer: {value}",
        ) from exc
    if result <= 0:
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            f"Jetson GPU resource bridge field {key} must be a positive integer: {value}",
        )
    return result


def _require_true(native_result: Any, key: str) -> None:
    value = _optional_value(native_result, key, None)
    if value is True:
        return
    raise TensorPreprocessError(
        JETSON_GPU_RESOURCE_BRIDGE_INVALID,
        f"Jetson GPU resource bridge result must include {key}=true",
    )


def _require_device_memory_space(native_result: Any) -> str:
    for key in JETSON_GPU_RESOURCE_BRIDGE_DEVICE_MEMORY_FIELDS:
        value = _optional_value(native_result, key, None)
        normalized = _normalize_capability_value(value)
        if not normalized:
            continue
        if normalized in JETSON_GPU_RESOURCE_BRIDGE_DEVICE_MEMORY_VALUES:
            return normalized
        raise TensorPreprocessError(
            JETSON_GPU_RESOURCE_BRIDGE_INVALID,
            "Jetson GPU resource bridge result must describe a device pointer; "
            f"{key}={value!r} is not accepted",
        )
    accepted = ", ".join(sorted(JETSON_GPU_RESOURCE_BRIDGE_DEVICE_MEMORY_VALUES))
    raise TensorPreprocessError(
        JETSON_GPU_RESOURCE_BRIDGE_INVALID,
        "Jetson GPU resource bridge result missing memory_space; accepted "
        f"values: {accepted}",
    )


def _optional_value(native_result: Any, key: str, default: Any) -> Any:
    if isinstance(native_result, Mapping):
        return native_result.get(key, default)
    return getattr(native_result, key, default)
