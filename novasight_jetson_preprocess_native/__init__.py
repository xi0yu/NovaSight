from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping


ABI_VERSION = 1
LIBRARY_ENV = "NOVASIGHT_JETSON_NATIVE_LIBRARY"
ABI_SYMBOL_ENV = "NOVASIGHT_JETSON_NATIVE_ABI_SYMBOL"
PREPARE_SYMBOL_ENV = "NOVASIGHT_JETSON_NATIVE_PREPARE_SYMBOL"
STATUS_SYMBOL_ENV = "NOVASIGHT_JETSON_NATIVE_STATUS_SYMBOL"
RELEASE_SYMBOL_ENV = "NOVASIGHT_JETSON_NATIVE_RELEASE_SYMBOL"
RESULT_BUFFER_BYTES_ENV = "NOVASIGHT_JETSON_NATIVE_RESULT_BUFFER_BYTES"
AUTO_BUILD_ENV = "NOVASIGHT_JETSON_NATIVE_AUTO_BUILD"
BUILD_DIR_ENV = "NOVASIGHT_JETSON_NATIVE_BUILD_DIR"
CMAKE_ENV = "NOVASIGHT_JETSON_NATIVE_CMAKE"
BUILD_TIMEOUT_S_ENV = "NOVASIGHT_JETSON_NATIVE_BUILD_TIMEOUT_S"
DEFAULT_ABI_SYMBOL = "novasight_abi_version"
DEFAULT_PREPARE_SYMBOL = "novasight_prepare_tensor_json"
DEFAULT_STATUS_SYMBOL = "novasight_status_json"
DEFAULT_RELEASE_SYMBOL = "novasight_release_tensor"
DEFAULT_RESULT_BUFFER_BYTES = 64 * 1024
DEFAULT_BUILD_TIMEOUT_S = 180.0
CANONICAL_LIBRARY_NAME = "libnovasight_preprocess.so"
LEGACY_LIBRARY_NAME = "libnovasight_jetson_preprocess_native.so"

CAPABILITIES = {
    "memory": ["nvmm", "dmabuf"],
    "resource_kind": ["gstreamer_sample"],
    "resource_source": ["appsink"],
    "formats": ["NV12"],
    "dtypes": ["float32", "float16"],
}

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
    "ctypes_library_env": LIBRARY_ENV,
    "ctypes_abi_symbol_env": ABI_SYMBOL_ENV,
    "ctypes_prepare_symbol_env": PREPARE_SYMBOL_ENV,
    "ctypes_status_symbol_env": STATUS_SYMBOL_ENV,
    "ctypes_release_symbol_env": RELEASE_SYMBOL_ENV,
    "ctypes_abi": "uint32_t abi_version(void)",
    "ctypes_prepare_abi": (
        "int prepare(const char* payload_json, char* result_json, "
        "size_t result_json_size)"
    ),
    "ctypes_status_abi": "int status(char* status_json, size_t status_json_size)",
    "ctypes_release_abi": "int release(uint64_t release_token)",
    "payload_json_notes": (
        "The ctypes bridge serializes JSON-safe payload fields only. "
        "resource_handle is not passed to C; a valid dmabuf_fd or gst_buffer_ptr "
        "is required. Python must keep the Gst.Sample/GstBuffer resource_handle "
        "alive until native preprocessing returns."
    ),
    "result_required_fields": [
        "device_ptr",
        "nbytes",
        "release_token",
        "zero_copy",
        "memory_space",
    ],
    "result_optional_fields": [
        "shape",
        "dtype",
        "stream",
        "backend",
        "reason",
    ],
    "result_semantics": (
        "Return an existing GPU device pointer containing a model-ready NCHW "
        "tensor. The native result must explicitly report zero_copy=true and "
        "memory_space=device/cuda/cuda_device/gpu/gpu_device. This module is "
        "unavailable until NOVASIGHT_JETSON_NATIVE_LIBRARY points at a Jetson "
        "shared library implementing the ctypes ABI."
    ),
}

_LIBRARY: Any | None = None
_LIBRARY_PATH = ""
_LIBRARY_ERROR = ""
_AUTO_BUILD_ATTEMPTED = False


def status() -> dict[str, Any]:
    library = _load_library()
    if library is None:
        reason = "native_implementation_missing"
        detail = (
            f"Set {LIBRARY_ENV} to a Jetson shared library that converts "
            "DMABUF/NvBufSurface/EGL/CUDA resources into a TensorRT DeviceTensor "
            f"({CANONICAL_LIBRARY_NAME})."
        )
        if _LIBRARY_ERROR:
            reason = "native_library_unavailable"
            detail = _LIBRARY_ERROR
        return {
            "available": False,
            "ready": False,
            "reason": reason,
            "detail": detail,
            "backend": "novasight_jetson_preprocess_native",
            "library": _library_path(),
            "required_abi_version": ABI_VERSION,
            "abi_version": None,
            "abi_compatible": False,
            "abi_symbol": _abi_symbol_name(),
            "prepare_symbol": _prepare_symbol_name(),
            "status_symbol": _status_symbol_name(),
            "release_symbol": _release_symbol_name(),
            "contract": CONTRACT,
        }
    prepare = _library_prepare(library)
    if prepare is None:
        return {
            "available": False,
            "ready": False,
            "reason": "native_prepare_symbol_missing",
            "detail": f"{_library_path()} does not export {_prepare_symbol_name()}",
            "backend": "novasight_jetson_preprocess_native",
            "library": _library_path(),
            "required_abi_version": ABI_VERSION,
            "abi_version": None,
            "abi_compatible": False,
            "abi_symbol": _abi_symbol_name(),
            "prepare_symbol": _prepare_symbol_name(),
            "status_symbol": _status_symbol_name(),
            "release_symbol": _release_symbol_name(),
            "contract": CONTRACT,
        }
    try:
        native_status = _call_status(library)
    except Exception as exc:
        return {
            "available": False,
            "ready": False,
            "reason": "native_status_failed",
            "detail": str(exc),
            "backend": "novasight_jetson_preprocess_native",
            "library": _library_path(),
            "required_abi_version": ABI_VERSION,
            "abi_version": None,
            "abi_compatible": False,
            "abi_symbol": _abi_symbol_name(),
            "prepare_symbol": _prepare_symbol_name(),
            "status_symbol": _status_symbol_name(),
            "release_symbol": _release_symbol_name(),
            "contract": CONTRACT,
        }
    abi_version, abi_reason, abi_detail = _library_abi_check(
        library,
        native_status=native_status,
    )
    if abi_reason:
        return {
            "available": False,
            "ready": False,
            "reason": abi_reason,
            "detail": abi_detail,
            "backend": "novasight_jetson_preprocess_native",
            "library": _library_path(),
            "required_abi_version": ABI_VERSION,
            "abi_version": abi_version,
            "abi_compatible": False,
            "abi_symbol": _abi_symbol_name(),
            "prepare_symbol": _prepare_symbol_name(),
            "status_symbol": _status_symbol_name(),
            "release_symbol": _release_symbol_name(),
            "contract": CONTRACT,
        }
    if native_status is not None:
        result = dict(native_status)
        result.setdefault("available", bool(result.get("ready", True)))
        result.setdefault("ready", bool(result.get("available", True)))
        result.setdefault("backend", "novasight_jetson_preprocess_native")
        result.setdefault("library", _library_path())
        result.setdefault("required_abi_version", ABI_VERSION)
        result.setdefault("abi_version", abi_version)
        result.setdefault("abi_compatible", True)
        result.setdefault("abi_symbol", _abi_symbol_name())
        result.setdefault("prepare_symbol", _prepare_symbol_name())
        result.setdefault("status_symbol", _status_symbol_name())
        result.setdefault("release_symbol", _release_symbol_name())
        result.setdefault("contract", CONTRACT)
        contract_error = _ready_output_contract_error(result)
        if contract_error:
            result["available"] = False
            result["ready"] = False
            result["reason"] = "native_output_contract_invalid"
            result["detail"] = contract_error
        return result
    return {
        "available": False,
        "ready": False,
        "backend": "novasight_jetson_preprocess_native",
        "library": _library_path(),
        "required_abi_version": ABI_VERSION,
        "abi_version": abi_version,
        "abi_compatible": True,
        "abi_symbol": _abi_symbol_name(),
        "prepare_symbol": _prepare_symbol_name(),
        "status_symbol": _status_symbol_name(),
        "release_symbol": _release_symbol_name(),
        "contract": CONTRACT,
        "reason": "native_readiness_status_required",
        "detail": (
            f"{_library_path()} must export {_status_symbol_name()} and report "
            "ready/available plus zero_copy=true and device memory_space before "
            "NVMM inference can start"
        ),
    }


def _probe_jetpack_component(
    *,
    header_paths: tuple[tuple[str, ...], ...],
    library_names: tuple[str, ...],
    apt_package: str,
) -> dict[str, Any]:
    """Inspect the filesystem to determine which JetPack component is missing."""
    found_header: str | None = None
    for candidates in header_paths:
        for candidate in candidates:
            if Path(candidate).is_file():
                found_header = candidate
                break
        if found_header is not None:
            break
    found_library: str | None = None
    for name in library_names:
        probes = (
            Path(f"/usr/lib/aarch64-linux-gnu/lib{name}.so"),
            Path(f"/usr/lib/aarch64-linux-gnu/lib{name}.so.1"),
            Path(f"/usr/lib/aarch64-linux-gnu/nvidia/lib{name}.so"),
            Path(f"/usr/lib/aarch64-linux-gnu/nvidia/lib{name}.so.1"),
            Path(f"/opt/nvidia/deepstream/deepstream-7.1/lib/lib{name}.so"),
            Path(f"/usr/lib/lib{name}.so"),
        )
        for probe in probes:
            if probe.is_file():
                found_library = str(probe)
                break
        if found_library is not None:
            break
    return {
        "header": found_header,
        "library": found_library,
        "apt_package": apt_package,
        "present": found_header is not None and found_library is not None,
    }


def preflight() -> dict[str, Any]:
    """Inspect Jetson-side dependencies required by the production build.

    Read-only; never invokes cmake. Returns a structured status object suitable
    for logging before attempting the auto-build. The output makes the precise
    JetPack apt package (and any of the standard search paths) auditable from
    a single command.
    """
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        return {
            "jetson_runtime": False,
            "platform": platform.system(),
            "machine": platform.machine(),
            "production_build_supported": False,
            "missing_components": [],
            "apt_install_command": "",
            "reason": "non_jetson_platform",
            "detail": (
                "Production Jetson preprocess can only be built on aarch64 Linux. "
                "Use NOVASIGHT_JETSON_NATIVE_LIBRARY to point to a pre-built "
                "libnovasight_preprocess.so when developing off-device."
            ),
        }
    nvbufsurface = _probe_jetpack_component(
        header_paths=(
            (
                "/usr/src/jetson_multimedia_api/include/nvbufsurface.h",
                "/usr/include/aarch64-linux-gnu/nvbufsurface.h",
                "/usr/include/nvbufsurface.h",
                "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvbufsurface.h",
            ),
        ),
        library_names=("nvbufsurface",),
        apt_package="nvidia-l4t-jetson-multimedia-api",
    )
    nvbufsurftransform = _probe_jetpack_component(
        header_paths=(
            (
                "/usr/src/jetson_multimedia_api/include/nvbufsurftransform.h",
                "/usr/include/aarch64-linux-gnu/nvbufsurftransform.h",
                "/usr/include/nvbufsurftransform.h",
                "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvbufsurftransform.h",
            ),
        ),
        library_names=("nvbufsurftransform",),
        apt_package="nvidia-l4t-jetson-multimedia-api",
    )
    egl = _probe_jetpack_component(
        header_paths=(
            (
                "/usr/include/EGL/egl.h",
                "/usr/include/aarch64-linux-gnu/EGL/egl.h",
            ),
        ),
        library_names=("EGL",),
        apt_package="libegl1 libegl-dev",
    )
    cuda_toolkit = _probe_jetpack_component(
        header_paths=(
            (
                "/usr/local/cuda/include/cuda.h",
                "/usr/include/cuda.h",
            ),
        ),
        library_names=("cudart",),
        apt_package="nvidia-cuda-toolkit-* (or JetPack cuda-toolkit)",
    )
    components = {
        "nvbufsurface": nvbufsurface,
        "nvbufsurftransform": nvbufsurftransform,
        "egl": egl,
        "cuda_toolkit": cuda_toolkit,
    }
    missing_components = [
        name for name, info in components.items() if not info["present"]
    ]
    apt_packages: list[str] = []
    seen: set[str] = set()
    for name in missing_components:
        pkg = components[name]["apt_package"]
        if pkg not in seen:
            apt_packages.append(pkg)
            seen.add(pkg)
    return {
        "jetson_runtime": True,
        "platform": "linux",
        "machine": platform.machine(),
        "production_build_supported": not missing_components,
        "components": components,
        "missing_components": missing_components,
        "apt_install_command": " ".join(["sudo apt install -y", *apt_packages]),
        "apt_packages": apt_packages,
        "reason": "production_build_unsupported" if missing_components else "production_build_ready",
        "detail": (
            "Missing JetPack components: " + ", ".join(missing_components) + ". "
            "Run the apt_install_command to enable the production build."
        )
        if missing_components
        else "",
    }



def prepare_tensor(payload: Mapping[str, Any]) -> Any:
    payload_json = _payload_json(payload)
    library = _load_library()
    if library is None:
        raise RuntimeError(status()["detail"])
    prepare = _library_prepare(library)
    if prepare is None:
        raise RuntimeError(f"{_library_path()} does not export {_prepare_symbol_name()}")
    try:
        native_status = _call_status(library)
    except Exception as exc:
        raise RuntimeError(f"native status failed: {exc}") from exc
    abi_version, abi_reason, abi_detail = _library_abi_check(
        library,
        native_status=native_status,
    )
    if abi_reason:
        raise RuntimeError(abi_detail)
    readiness_error = _native_readiness_error(
        native_status=native_status,
        abi_version=abi_version,
    )
    if readiness_error:
        raise RuntimeError(readiness_error)
    result_json = _call_json_function(prepare, payload_json)
    result = json.loads(result_json)
    if not isinstance(result, dict):
        raise RuntimeError("native prepare result JSON must be an object")
    _validate_prepare_result_contract(result)
    release = _library_release(library)
    release_token = result.get("release_token")
    if release_token is not None and release is None:
        raise RuntimeError(
            "native prepare result returned release_token but "
            f"{_release_symbol_name()} is not exported"
        )
    if release is not None and release_token is not None:
        result.setdefault("owner", _NativeTensorOwner(release, int(release_token)))
    result.setdefault("backend", "novasight_jetson_preprocess_native:ctypes")
    return result


def prepare_nvmm_tensor(payload: Mapping[str, Any]) -> Any:
    return prepare_tensor(payload)


def _payload_json(payload: Mapping[str, Any]) -> bytes:
    _validate_payload_contract(payload)
    dmabuf_fd = _optional_nonnegative_int(payload.get("dmabuf_fd"))
    resource_metadata = _json_safe(payload.get("resource_metadata", {}))
    gst_buffer_ptr = _optional_positive_int(payload.get("gst_buffer_ptr"))
    if gst_buffer_ptr is None and isinstance(resource_metadata, Mapping):
        gst_buffer_ptr = _optional_positive_int(resource_metadata.get("gst_buffer_ptr"))
    serializable = {
        "frame_id": _required_positive_int(payload, "frame_id"),
        "capture_ts_ns": _required_positive_int(payload, "capture_ts_ns"),
        "resource_kind": _required_text(payload, "resource_kind"),
        "resource_memory": _required_text(payload, "resource_memory"),
        "resource_source": _required_text(payload, "resource_source"),
        "dmabuf_fd": dmabuf_fd,
        "gst_buffer_ptr": gst_buffer_ptr,
        "resource_metadata": resource_metadata,
        "resource_width": _required_positive_int(payload, "resource_width"),
        "resource_height": _required_positive_int(payload, "resource_height"),
        "resource_pixel_format": _required_text(payload, "resource_pixel_format"),
        "width": _required_positive_int(payload, "width"),
        "height": _required_positive_int(payload, "height"),
        "pixel_format": _required_text(payload, "pixel_format"),
        "source_width": _required_positive_int(payload, "source_width"),
        "source_height": _required_positive_int(payload, "source_height"),
        "roi_offset_x": _required_nonnegative_int(payload, "roi_offset_x"),
        "roi_offset_y": _required_nonnegative_int(payload, "roi_offset_y"),
        "needs_resize": _required_bool(payload, "needs_resize"),
        "model_shape": _json_safe(_required_mapping(payload, "model_shape")),
        "nchw": _required_nchw(payload),
        "dtype": _required_text(payload, "dtype"),
    }
    return json.dumps(serializable, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _validate_payload_contract(payload: Mapping[str, Any]) -> None:
    required = CONTRACT["payload_required_fields"]
    missing = [field for field in required if field != "resource_handle" and field not in payload]
    if missing:
        raise RuntimeError(
            "ctypes Jetson native backend payload missing required field(s): "
            + ", ".join(missing)
        )
    _required_positive_int(payload, "frame_id")
    _required_positive_int(payload, "capture_ts_ns")
    if _required_text(payload, "resource_kind") != "gstreamer_sample":
        raise RuntimeError("ctypes Jetson native backend requires resource_kind=gstreamer_sample")
    if _required_text(payload, "resource_memory").lower() not in {"nvmm", "dmabuf"}:
        raise RuntimeError("ctypes Jetson native backend requires resource_memory=nvmm or dmabuf")
    if _required_text(payload, "resource_source") != "appsink":
        raise RuntimeError("ctypes Jetson native backend requires resource_source=appsink")
    if not isinstance(payload.get("resource_metadata"), Mapping):
        raise RuntimeError("ctypes Jetson native backend requires resource_metadata to be an object")
    dmabuf_fd = _optional_nonnegative_int(payload.get("dmabuf_fd"))
    gst_buffer_ptr = _optional_positive_int(payload.get("gst_buffer_ptr"))
    if gst_buffer_ptr is None:
        metadata = payload.get("resource_metadata")
        if isinstance(metadata, Mapping):
            gst_buffer_ptr = _optional_positive_int(metadata.get("gst_buffer_ptr"))
    if dmabuf_fd is None and gst_buffer_ptr is None:
        raise RuntimeError(
            "ctypes Jetson native backend requires dmabuf_fd or gst_buffer_ptr"
        )
    resource_width = _required_positive_int(payload, "resource_width")
    resource_height = _required_positive_int(payload, "resource_height")
    if _required_text(payload, "resource_pixel_format").upper() != "NV12":
        raise RuntimeError("ctypes Jetson native backend requires resource_pixel_format=NV12")
    _required_positive_int(payload, "width")
    _required_positive_int(payload, "height")
    if _required_text(payload, "pixel_format").upper() != "NV12":
        raise RuntimeError("ctypes Jetson native backend requires pixel_format=NV12")
    if resource_width != int(payload["width"]):
        raise RuntimeError("ctypes Jetson native backend requires resource_width to match width")
    if resource_height != int(payload["height"]):
        raise RuntimeError("ctypes Jetson native backend requires resource_height to match height")
    _required_positive_int(payload, "source_width")
    _required_positive_int(payload, "source_height")
    _required_nonnegative_int(payload, "roi_offset_x")
    _required_nonnegative_int(payload, "roi_offset_y")
    _required_bool(payload, "needs_resize")
    model_shape = _required_mapping(payload, "model_shape")
    for key in ("batch", "channels", "height", "width"):
        _required_positive_int(model_shape, key, prefix="model_shape.")
    nchw = _required_nchw(payload)
    if nchw[1] not in {1, 3, 4}:
        raise RuntimeError("ctypes Jetson native backend requires nchw channels to be 1, 3, or 4")
    if _required_text(payload, "dtype") not in {"float32", "float16"}:
        raise RuntimeError("ctypes Jetson native backend requires dtype=float32 or float16")


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise RuntimeError(f"ctypes Jetson native backend requires non-empty {key}")
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"ctypes Jetson native backend requires {key} to be boolean")
    return value


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"ctypes Jetson native backend requires {key} to be an object")
    return value


def _required_positive_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> int:
    try:
        value = int(payload.get(key))
    except Exception as exc:
        raise RuntimeError(
            f"ctypes Jetson native backend requires {prefix}{key} to be a positive integer"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"ctypes Jetson native backend requires {prefix}{key} to be a positive integer"
        )
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = int(payload.get(key))
    except Exception as exc:
        raise RuntimeError(
            f"ctypes Jetson native backend requires {key} to be a non-negative integer"
        ) from exc
    if value < 0:
        raise RuntimeError(
            f"ctypes Jetson native backend requires {key} to be a non-negative integer"
        )
    return value


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def _optional_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _required_nchw(payload: Mapping[str, Any]) -> list[int]:
    raw = payload.get("nchw")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise RuntimeError("ctypes Jetson native backend requires nchw=[N,C,H,W]")
    values = []
    for index, item in enumerate(raw):
        try:
            value = int(item)
        except Exception as exc:
            raise RuntimeError(
                f"ctypes Jetson native backend requires nchw[{index}] to be a positive integer"
            ) from exc
        if value <= 0:
            raise RuntimeError(
                f"ctypes Jetson native backend requires nchw[{index}] to be a positive integer"
            )
        values.append(value)
    return values


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_json_safe(item) for item in value]
        return str(value)
    return value


def _call_status(library: Any) -> dict[str, Any] | None:
    status_fn = _library_symbol(library, _status_symbol_name())
    if status_fn is None:
        return None
    status_fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    status_fn.restype = ctypes.c_int
    result = _call_json_function(status_fn, b"")
    value = json.loads(result)
    if not isinstance(value, dict):
        raise RuntimeError("native status result JSON must be an object")
    return value


def _library_abi_check(
    library: Any,
    *,
    native_status: Mapping[str, Any] | None,
) -> tuple[int | None, str, str]:
    versions: list[tuple[str, int]] = []
    if isinstance(native_status, Mapping) and native_status.get("abi_version") is not None:
        try:
            versions.append(("status", int(native_status["abi_version"])))
        except Exception:
            return (
                None,
                "native_abi_incompatible",
                f"{_library_path()} status abi_version is not an integer: "
                f"{native_status.get('abi_version')!r}",
            )
    abi_fn = _library_symbol(library, _abi_symbol_name())
    if abi_fn is not None:
        try:
            abi_fn.argtypes = []
            abi_fn.restype = ctypes.c_uint32
            versions.append(("symbol", int(abi_fn())))
        except Exception as exc:
            return (
                None,
                "native_abi_incompatible",
                f"{_library_path()} {_abi_symbol_name()} failed: {exc}",
            )
    if not versions:
        return (
            None,
            "native_abi_version_missing",
            f"{_library_path()} must export {_abi_symbol_name()} or report "
            f"abi_version={ABI_VERSION} from {_status_symbol_name()}",
        )
    mismatched = [
        f"{source}={version}" for source, version in versions if version != ABI_VERSION
    ]
    if mismatched:
        return (
            versions[0][1],
            "native_abi_incompatible",
            f"{_library_path()} ABI mismatch: expected {ABI_VERSION}, got "
            + ", ".join(mismatched),
        )
    distinct = {version for _, version in versions}
    if len(distinct) > 1:
        return (
            versions[0][1],
            "native_abi_incompatible",
            f"{_library_path()} reports inconsistent ABI versions: {versions}",
        )
    return versions[0][1], "", ""


def _call_json_function(function: Any, payload_json: bytes) -> str:
    buffer = ctypes.create_string_buffer(_result_buffer_bytes())
    if payload_json:
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        rc = int(function(ctypes.c_char_p(payload_json), buffer, ctypes.sizeof(buffer)))
    else:
        rc = int(function(buffer, ctypes.sizeof(buffer)))
    result = buffer.value.decode("utf-8", errors="replace")
    if rc != 0:
        raise RuntimeError(_native_error_detail(result, rc=rc))
    if not result:
        raise RuntimeError("native function returned an empty JSON result")
    return result


def _native_error_detail(result: str, *, rc: int) -> str:
    text = result.strip()
    if not text:
        return f"native function returned rc={rc}"
    try:
        payload = json.loads(text)
    except Exception:
        return text
    if not isinstance(payload, Mapping):
        return text
    reason = str(payload.get("reason") or "").strip()
    detail = str(payload.get("detail") or "").strip()
    if reason == "dmabuf_fd_required":
        return (
            "loaded Jetson native preprocess library is stale: it still requires "
            "dmabuf_fd and does not support the GstBuffer pointer fallback. "
            f"library={_library_path()}; rebuild or replace the native shared "
            "library from the current source before starting NVMM inference. "
            f"native_detail={detail or text}"
        )
    return text


def _validate_prepare_result_contract(result: Mapping[str, Any]) -> None:
    for key in ("device_ptr", "nbytes", "release_token"):
        value = result.get(key)
        try:
            parsed = int(value)
        except Exception as exc:
            raise RuntimeError(
                f"native prepare result field {key} must be a positive integer: {value!r}"
            ) from exc
        if parsed <= 0:
            raise RuntimeError(
                f"native prepare result field {key} must be a positive integer: {value!r}"
            )
    if result.get("zero_copy") is not True:
        raise RuntimeError("native prepare result must include zero_copy=true")
    memory_space = str(result.get("memory_space", "")).strip().lower().replace("-", "_")
    accepted = {"device", "cuda", "cuda_device", "gpu", "gpu_device"}
    if memory_space not in accepted:
        raise RuntimeError(
            "native prepare result must include memory_space=device/cuda/cuda_device/"
            f"gpu/gpu_device, got {result.get('memory_space')!r}"
        )


def _native_readiness_error(
    *,
    native_status: Mapping[str, Any] | None,
    abi_version: int | None,
) -> str:
    if native_status is None:
        return (
            f"{_library_path()} must export {_status_symbol_name()} and report "
            "ready/available plus zero_copy=true and device memory_space before "
            "prepare can be called"
        )
    status_payload = dict(native_status)
    status_payload.setdefault("available", bool(status_payload.get("ready", True)))
    status_payload.setdefault("ready", bool(status_payload.get("available", True)))
    status_payload.setdefault("backend", "novasight_jetson_preprocess_native")
    status_payload.setdefault("abi_version", abi_version)
    if not (
        bool(status_payload.get("available", False))
        or bool(status_payload.get("ready", False))
    ):
        detail = status_payload.get("detail") or status_payload.get("reason")
        return str(detail or "native library is not ready")
    return _ready_output_contract_error(status_payload)


def _ready_output_contract_error(status_payload: Mapping[str, Any]) -> str:
    if not (
        bool(status_payload.get("available", False))
        or bool(status_payload.get("ready", False))
    ):
        return ""
    backend = str(status_payload.get("backend", "")).lower()
    if "reference" in backend or "scaffold" in backend:
        return f"backend={status_payload.get('backend')!r} is not a production Jetson backend"
    if status_payload.get("zero_copy") is not True:
        return "ready native library status must include zero_copy=true"
    memory_space = str(status_payload.get("memory_space", "")).strip().lower().replace("-", "_")
    accepted = {"device", "cuda", "cuda_device", "gpu", "gpu_device"}
    if memory_space not in accepted:
        return (
            "ready native library status must include memory_space=device/"
            f"cuda/cuda_device/gpu/gpu_device, got {status_payload.get('memory_space')!r}"
        )
    return ""


def _load_library() -> Any | None:
    global _LIBRARY, _LIBRARY_ERROR, _LIBRARY_PATH
    path = _library_path()
    if _LIBRARY is not None and _LIBRARY_PATH == path:
        return _LIBRARY
    if _LIBRARY_PATH != path:
        _LIBRARY = None
        _LIBRARY_ERROR = ""
        _LIBRARY_PATH = path
    if not path:
        return None
    try:
        _LIBRARY = ctypes.CDLL(path)
    except Exception as exc:
        _LIBRARY_ERROR = f"{path}: {exc}"
        return None
    _LIBRARY_ERROR = ""
    return _LIBRARY


def _reset_library_cache() -> None:
    global _LIBRARY, _LIBRARY_ERROR, _LIBRARY_PATH, _AUTO_BUILD_ATTEMPTED
    _LIBRARY = None
    _LIBRARY_ERROR = ""
    _LIBRARY_PATH = ""
    _AUTO_BUILD_ATTEMPTED = False


def _library_prepare(library: Any) -> Any | None:
    return _library_symbol(library, _prepare_symbol_name())


def _library_release(library: Any) -> Any | None:
    release = _library_symbol(library, _release_symbol_name())
    if release is not None:
        release.argtypes = [ctypes.c_uint64]
        release.restype = ctypes.c_int
    return release


def _library_symbol(library: Any, name: str) -> Any | None:
    try:
        return getattr(library, name)
    except AttributeError:
        return None


def _library_path() -> str:
    configured = os.environ.get(LIBRARY_ENV, "").strip()
    if configured:
        return configured
    for candidate in _default_library_candidates():
        if candidate.is_file():
            return str(candidate)
    built = _ensure_default_library_built()
    if built is not None and built.is_file():
        return str(built)
    return ""


def _default_library_candidates() -> list[Path]:
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parent
    cwd = Path.cwd()
    build_dirs = [
        cwd / "build/jetson-native",
        cwd / "build" / "jetson-native",
        repo_root / "build/jetson-native",
        repo_root / "build" / "jetson-native",
    ]
    candidates = [
        build_dir / library_name
        for build_dir in build_dirs
        for library_name in (CANONICAL_LIBRARY_NAME, LEGACY_LIBRARY_NAME)
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _ensure_default_library_built() -> Path | None:
    global _AUTO_BUILD_ATTEMPTED, _LIBRARY_ERROR
    if _AUTO_BUILD_ATTEMPTED:
        return None
    _AUTO_BUILD_ATTEMPTED = True
    if not _auto_build_enabled():
        return None
    if not _is_jetson_runtime():
        return None

    package_dir = Path(__file__).resolve().parent
    native_dir = package_dir / "native"
    production_source = native_dir / "src/jetson/novasight_jetson_preprocess_native_jetson_cuda.cu"
    build_dir = _default_build_dir()
    library = build_dir / CANONICAL_LIBRARY_NAME
    legacy_library = build_dir / LEGACY_LIBRARY_NAME
    cmake = os.environ.get(CMAKE_ENV, "cmake").strip() or "cmake"
    timeout_s = _auto_build_timeout_s()

    if not native_dir.is_dir():
        _LIBRARY_ERROR = f"Jetson native source tree missing: {native_dir}"
        return None
    if not production_source.is_file():
        _LIBRARY_ERROR = f"Jetson native production source missing: {production_source}"
        return None

    configure_command = [
        cmake,
        "-S",
        str(native_dir),
        "-B",
        str(build_dir),
        "-DNOVASIGHT_JETSON_PREPROCESS_IMPL=jetson",
        f"-DNOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE={production_source}",
    ]
    build_command = [cmake, "--build", str(build_dir)]

    try:
        configure = _run_auto_build_command(configure_command, timeout_s)
    except Exception as exc:
        _LIBRARY_ERROR = (
            "Jetson native auto-build configure failed before completion "
            f"command={' '.join(configure_command)}; error={exc}"
        )
        return None
    if int(getattr(configure, "returncode", 1)) != 0:
        _LIBRARY_ERROR = _command_failure_detail("configure", configure_command, configure)
        return None
    try:
        build = _run_auto_build_command(build_command, timeout_s)
    except Exception as exc:
        _LIBRARY_ERROR = (
            "Jetson native auto-build build failed before completion "
            f"command={' '.join(build_command)}; error={exc}"
        )
        return None
    if int(getattr(build, "returncode", 1)) != 0:
        _LIBRARY_ERROR = _command_failure_detail("build", build_command, build)
        return None
    if not library.is_file() and legacy_library.is_file():
        library = legacy_library
    if not library.is_file():
        _LIBRARY_ERROR = (
            "Jetson native build completed but library is missing: "
            f"{build_dir / CANONICAL_LIBRARY_NAME}"
        )
        return None
    _LIBRARY_ERROR = ""
    return library


def _default_build_dir() -> Path:
    configured = os.environ.get(BUILD_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "build" / "jetson-native"


def _auto_build_enabled() -> bool:
    value = os.environ.get(AUTO_BUILD_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disable", "disabled"}


def _auto_build_timeout_s() -> float:
    try:
        return max(5.0, float(os.environ.get(BUILD_TIMEOUT_S_ENV, DEFAULT_BUILD_TIMEOUT_S)))
    except Exception:
        return DEFAULT_BUILD_TIMEOUT_S


def _is_jetson_runtime() -> bool:
    if platform.system() != "Linux":
        return False
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return False
    jetson_markers = (
        Path("/etc/nv_tegra_release"),
        Path("/usr/src/jetson_multimedia_api/include/nvbufsurface.h"),
        Path("/usr/include/aarch64-linux-gnu/nvbufsurface.h"),
        Path("/usr/include/nvbufsurface.h"),
    )
    return any(marker.exists() for marker in jetson_markers)


def _run_auto_build_command(command: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _command_failure_detail(
    phase: str,
    command: list[str],
    result: Any,
) -> str:
    output = "\n".join(
        item
        for item in (
            str(getattr(result, "stderr", "") or "").strip(),
            str(getattr(result, "stdout", "") or "").strip(),
        )
        if item
    )
    if len(output) > 2000:
        output = output[-2000:]
    detail = (
        f"Jetson native auto-build {phase} failed rc={getattr(result, 'returncode', '?')} "
        f"command={' '.join(command)}"
        + (f"; output={output}" if output else "")
    )
    if phase == "configure" and _is_jetson_runtime():
        try:
            check = preflight()
        except Exception:
            check = None
        if isinstance(check, dict) and check.get("missing_components"):
            detail = (
                detail
                + " | preflight_missing="
                + ",".join(check["missing_components"])
                + " | apt_fix=\""
                + str(check.get("apt_install_command") or "")
                + "\""
            )
    return detail


def _abi_symbol_name() -> str:
    return os.environ.get(ABI_SYMBOL_ENV, DEFAULT_ABI_SYMBOL).strip()


def _prepare_symbol_name() -> str:
    return os.environ.get(PREPARE_SYMBOL_ENV, DEFAULT_PREPARE_SYMBOL).strip()


def _status_symbol_name() -> str:
    return os.environ.get(STATUS_SYMBOL_ENV, DEFAULT_STATUS_SYMBOL).strip()


def _release_symbol_name() -> str:
    return os.environ.get(RELEASE_SYMBOL_ENV, DEFAULT_RELEASE_SYMBOL).strip()


def _result_buffer_bytes() -> int:
    try:
        value = int(os.environ.get(RESULT_BUFFER_BYTES_ENV, DEFAULT_RESULT_BUFFER_BYTES))
    except Exception:
        return DEFAULT_RESULT_BUFFER_BYTES
    return max(1024, value)


class _NativeTensorOwner:
    def __init__(self, release: Any, token: int) -> None:
        self._release = release
        self._token = int(token)
        self._released = False

    @property
    def release_token(self) -> int:
        return self._token

    def release(self) -> None:
        if self._released:
            return
        rc = int(self._release(self._token))
        if rc != 0:
            raise RuntimeError(f"native release failed for token={self._token} rc={rc}")
        self._released = True

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
