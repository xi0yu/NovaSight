from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any, Mapping


ABI_VERSION = 1
NATIVE_BACKEND_ENV = "NOVASIGHT_JETSON_NATIVE_PREPROCESSOR"
DEFAULT_NATIVE_BACKEND = "novasight_jetson_preprocess_native"
CONTRACT = {
    "abi_version": ABI_VERSION,
    "callable": "prepare_tensor(payload) or prepare_nvmm_tensor(payload)",
    "payload_required_fields": [
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
    ],
    "result_required_fields": ["device_ptr", "nbytes", "zero_copy", "memory_space"],
    "result_optional_fields": [
        "shape",
        "dtype",
        "stream",
        "owner",
        "backend",
        "reason",
    ],
    "result_semantics": (
        "Return an existing GPU device pointer containing a model-ready NCHW "
        "tensor. Successful native readiness must explicitly report "
        "zero_copy=true and memory_space=device/cuda/cuda_device/gpu/gpu_device. "
        "Do not report successful inference readiness without a real "
        "NVMM/DMABUF/NvBufSurface/EGL/CUDA conversion path."
    ),
}

CAPABILITIES = {
    "memory": ["nvmm", "dmabuf"],
    "resource_kind": ["gstreamer_sample"],
    "resource_source": ["appsink"],
    "formats": ["NV12"],
    "dtypes": ["float32", "float16"],
}

_BACKEND: ModuleType | None = None
_BACKEND_NAME = ""
_BACKEND_ERROR = ""


def status() -> dict[str, Any]:
    backend = _load_backend()
    if backend is None:
        return {
            "available": False,
            "ready": False,
            "reason": "native_backend_unavailable",
            "detail": _BACKEND_ERROR,
            "backend": _backend_module_name(),
            "env": NATIVE_BACKEND_ENV,
            "contract": CONTRACT,
        }
    backend_status = _backend_status(backend)
    if backend_status is not None:
        result = dict(backend_status)
        result.setdefault("backend", _backend_module_name())
        result.setdefault("env", NATIVE_BACKEND_ENV)
        result.setdefault("contract", CONTRACT)
        if "available" not in result and "ready" not in result:
            result["available"] = False
            result["ready"] = False
            result["reason"] = "native_readiness_status_required"
            result["detail"] = (
                "native backend status must explicitly report available or ready"
            )
        return result
    return {
        "available": False,
        "ready": False,
        "backend": _backend_module_name(),
        "env": NATIVE_BACKEND_ENV,
        "contract": CONTRACT,
        "reason": "native_readiness_status_required",
        "detail": (
            "native backend must expose status(), bridge_status(), get_status(), "
            "or get_bridge_status() before NVMM inference can start"
        ),
    }


def prepare_tensor(payload: Mapping[str, Any]) -> Any:
    backend = _load_backend()
    if backend is None:
        raise RuntimeError(_BACKEND_ERROR or "Jetson native preprocessing backend is unavailable")
    prepare = _backend_prepare(backend)
    if prepare is None:
        raise RuntimeError(f"{_backend_module_name()} must expose prepare_tensor(payload)")
    return prepare(dict(payload))


def prepare_nvmm_tensor(payload: Mapping[str, Any]) -> Any:
    return prepare_tensor(payload)


def _backend_module_name() -> str:
    return os.environ.get(NATIVE_BACKEND_ENV, DEFAULT_NATIVE_BACKEND).strip()


def _load_backend() -> ModuleType | None:
    global _BACKEND, _BACKEND_ERROR, _BACKEND_NAME
    module_name = _backend_module_name()
    if _BACKEND is not None and _BACKEND_NAME == module_name:
        return _BACKEND
    if _BACKEND_NAME != module_name:
        _BACKEND = None
        _BACKEND_ERROR = ""
        _BACKEND_NAME = module_name
    if not module_name:
        _BACKEND_ERROR = f"{NATIVE_BACKEND_ENV} is empty"
        return None
    try:
        _BACKEND = importlib.import_module(module_name)
    except Exception as exc:
        _BACKEND_ERROR = f"{module_name}: {exc}"
        return None
    _BACKEND_ERROR = ""
    return _BACKEND


def _reset_backend_cache() -> None:
    global _BACKEND, _BACKEND_ERROR, _BACKEND_NAME
    _BACKEND = None
    _BACKEND_NAME = ""
    _BACKEND_ERROR = ""


def _backend_status(backend: ModuleType) -> Mapping[str, Any] | None:
    for name in ("status", "bridge_status", "get_status", "get_bridge_status"):
        candidate = getattr(backend, name, None)
        if not callable(candidate):
            continue
        value = candidate()
        if isinstance(value, Mapping):
            return value
        return {"available": bool(value)}
    return None


def _backend_prepare(backend: ModuleType) -> Any | None:
    for name in ("prepare_tensor", "prepare_nvmm_tensor"):
        candidate = getattr(backend, name, None)
        if callable(candidate):
            return candidate
    if callable(backend):
        return backend
    return None
