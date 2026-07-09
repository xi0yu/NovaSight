from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .pipeline import CaptureCandidate
from .state import CaptureProfile


FrameResourceMemory = Literal["cpu", "gstreamer", "nvmm", "dmabuf", "cuda", "unknown"]


@dataclass(frozen=True)
class FrameResource:
    """Opaque frame resource carried with CaptureFrame/RoiFrame.

    `image` remains the CPU view used by the current inference fallback. This
    resource is the ownership hook for NVMM/DMABUF/CUDA-style paths and should
    not be interpreted as pixel data by generic runtime code.
    """

    kind: str
    handle: Any
    memory: FrameResourceMemory
    width: int
    height: int
    pixel_format: str
    source: str = ""
    dmabuf_fd: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gpu_accessible(self) -> bool:
        return self.memory in {"nvmm", "dmabuf", "cuda"}

    @property
    def gst_buffer_ptr(self) -> int | None:
        return _valid_pointer(self.metadata.get("gst_buffer_ptr"))


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    width: int
    height: int
    pixel_format: str
    ts_ns: int
    capture_wait_ms: float
    image: Any
    frame_resource: FrameResource | None = None
    userspace_process_ms: float = 0.0
    source_ts_ns: int | None = None
    source_ts_kind: str = ""
    source_width: int | None = None
    source_height: int | None = None
    roi_size: int | None = None
    roi_offset_x: int = 0
    roi_offset_y: int = 0

    @property
    def capture_ts_ns(self) -> int:
        """Design-contract name for the monotonic timestamp attached at capture."""
        return self.ts_ns

    @property
    def capture_ts_source(self) -> str:
        return "userspace_monotonic_receive"

    @property
    def receive_ts_ns(self) -> int:
        """Application monotonic timestamp captured when the frame reached userspace."""
        return self.ts_ns

    @property
    def capture_width(self) -> int:
        return int(self.source_width or self.width)

    @property
    def capture_height(self) -> int:
        return int(self.source_height or self.height)

    @property
    def roi_x(self) -> int:
        return int(self.roi_offset_x)

    @property
    def roi_y(self) -> int:
        return int(self.roi_offset_y)

    @property
    def roi_width(self) -> int:
        return int(self.roi_size or self.width)

    @property
    def roi_height(self) -> int:
        return int(self.roi_size or self.height)

    @property
    def image_ref(self) -> Any:
        if self.image is not None:
            return self.image
        if self.frame_resource is not None:
            return self.frame_resource.handle
        return None

    @property
    def gpu_buffer(self) -> Any | None:
        if self.frame_resource is not None and self.frame_resource.gpu_accessible:
            return self.frame_resource.handle
        return None

    @property
    def resource_memory(self) -> str:
        return self.frame_resource.memory if self.frame_resource is not None else "cpu"

    @property
    def dmabuf_fd(self) -> int | None:
        return self.frame_resource.dmabuf_fd if self.frame_resource is not None else None

    @property
    def gst_buffer_ptr(self) -> int | None:
        if self.frame_resource is None:
            return None
        return self.frame_resource.gst_buffer_ptr


CaptureFrame = CapturedFrame


class FrameSource(Protocol):
    backend_label: str

    def read(self) -> CapturedFrame | None: ...
    def close(self) -> None: ...


class ImageFrameSource:
    backend_label = "image:file"

    def __init__(self, path: str, *, fps: int = 30) -> None:
        from PIL import Image

        image_path = str(path).strip()
        if not image_path:
            raise ValueError("image path is required")
        self.path = image_path
        self.fps = max(1, int(fps))
        self._image = Image.open(self.path).convert("RGB")
        self._frame_id = 0
        self._closed = False
        self._last_ts_ns: int | None = None

    @property
    def width(self) -> int:
        return int(self._image.width)

    @property
    def height(self) -> int:
        return int(self._image.height)

    def read(self) -> CapturedFrame | None:
        if self._closed:
            return None
        t0 = time.monotonic_ns()
        if self._last_ts_ns is not None:
            interval_ns = int(1_000_000_000 / self.fps)
            sleep_ns = interval_ns - (t0 - self._last_ts_ns)
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
        t1 = time.monotonic_ns()
        self._last_ts_ns = t1
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            width=self.width,
            height=self.height,
            pixel_format="RGB",
            ts_ns=t1,
            capture_wait_ms=(t1 - t0) / 1e6,
            image=self._image.copy(),
            source_ts_ns=t1,
            source_ts_kind="synthetic_monotonic",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._image.close()
        self._closed = True


class OpenCvFrameSource:
    @classmethod
    def probe(cls, profile: CaptureProfile, candidate: CaptureCandidate) -> bool:
        source = cls(profile, candidate)
        try:
            return source.opened_and_readable()
        finally:
            source.close()

    def __init__(self, profile: CaptureProfile, candidate: CaptureCandidate) -> None:
        import cv2

        self.profile = profile
        self._candidate = candidate
        self.backend_label = candidate.label
        self._cv2 = cv2
        if candidate.label == "opencv:v4l2":
            self._cap = cv2.VideoCapture(profile.device, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, profile.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, profile.height)
            self._cap.set(cv2.CAP_PROP_FPS, profile.fps)
        else:
            self._cap = cv2.VideoCapture(candidate.pipeline, cv2.CAP_GSTREAMER)
        self._frame_id = 0
        self._closed = False

    def opened_and_readable(self) -> bool:
        if not self._cap.isOpened():
            self.close()
            return False
        ok, image = self._cap.read()
        if not ok or image is None:
            self.close()
            return False
        return True

    def read(self) -> CapturedFrame | None:
        t0 = time.monotonic_ns()
        ok, image = self._cap.read()
        t1 = time.monotonic_ns()
        if not ok or image is None:
            return None
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            pixel_format="BGR",
            ts_ns=t1,
            capture_wait_ms=(t1 - t0) / 1e6,
            image=image,
            source_ts_ns=None,
            source_ts_kind="",
            source_width=self._candidate.source_width,
            source_height=self._candidate.source_height,
            roi_size=self._candidate.roi_size,
            roi_offset_x=self._candidate.roi_offset_x,
            roi_offset_y=self._candidate.roi_offset_y,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._cap.release()
        self._closed = True


class GstAppSinkFrameSource:
    @classmethod
    def probe(cls, profile: CaptureProfile, candidate: CaptureCandidate) -> bool:
        source = cls(profile, candidate)
        try:
            return source.opened_and_readable()
        finally:
            source.close()

    def __init__(self, profile: CaptureProfile, candidate: CaptureCandidate) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst, GstApp  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"PyGObject Gst/GstApp unavailable: {exc}") from exc

        if not Gst.is_initialized():
            Gst.init(None)
        self.profile = profile
        self._candidate = candidate
        self.backend_label = candidate.label
        self._Gst = Gst
        self._pipeline = Gst.parse_launch(candidate.pipeline)
        self._closed = False
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            self._stop_pipeline()
            raise RuntimeError("appsink element not found")
        self._frame_id = 0
        self._opened_readable = False
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            diagnostics = _drain_bus_diagnostics(self._pipeline, Gst)
            self._stop_pipeline()
            message = "GStreamer pipeline set_state PLAYING failed"
            if diagnostics:
                message = f"{message}; bus diagnostics: {'; '.join(diagnostics)}"
            raise RuntimeError(message)
        try:
            first = self._pull_frame(timeout_ns=2 * Gst.SECOND)
            if first is None:
                diagnostics = _drain_bus_diagnostics(self._pipeline, Gst)
                message = "GStreamer appsink first frame timeout"
                if diagnostics:
                    message = f"{message}; bus diagnostics: {'; '.join(diagnostics)}"
                raise RuntimeError(message)
        except Exception:
            self._stop_pipeline()
            raise
        self._frame_id = 0
        self._opened_readable = True

    def opened_and_readable(self) -> bool:
        return bool(self._opened_readable)

    def read(self) -> CapturedFrame | None:
        return self._pull_frame(timeout_ns=100 * self._Gst.MSECOND)

    def close(self) -> None:
        if self._closed:
            return
        self._stop_pipeline()

    def _stop_pipeline(self) -> None:
        self._pipeline.set_state(self._Gst.State.NULL)
        self._pipeline.get_state(2 * self._Gst.SECOND)
        self._closed = True

    def _pull_frame(self, *, timeout_ns: int) -> CapturedFrame | None:
        t0 = time.monotonic_ns()
        sample = self._appsink.try_pull_sample(timeout_ns)
        receive_ts_ns = time.monotonic_ns()
        if sample is None:
            return None
        source_ts_ns, source_ts_kind = _sample_source_timestamp(sample, self._Gst)
        image, width, height, fmt = _sample_to_bgr(sample, self._Gst)
        frame_resource = _sample_frame_resource(
            sample,
            self._Gst,
            width=width,
            height=height,
            pixel_format=fmt,
        )
        ready_ts_ns = time.monotonic_ns()
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            width=width,
            height=height,
            pixel_format="BGR",
            ts_ns=receive_ts_ns,
            capture_wait_ms=(receive_ts_ns - t0) / 1e6,
            image=image,
            frame_resource=frame_resource,
            userspace_process_ms=(ready_ts_ns - receive_ts_ns) / 1e6,
            source_ts_ns=source_ts_ns,
            source_ts_kind=source_ts_kind,
            source_width=self._candidate.source_width,
            source_height=self._candidate.source_height,
            roi_size=self._candidate.roi_size,
            roi_offset_x=self._candidate.roi_offset_x,
            roi_offset_y=self._candidate.roi_offset_y,
        )


class GstResourceFrameSource(GstAppSinkFrameSource):
    def _pull_frame(self, *, timeout_ns: int) -> CapturedFrame | None:
        t0 = time.monotonic_ns()
        sample = self._appsink.try_pull_sample(timeout_ns)
        receive_ts_ns = time.monotonic_ns()
        if sample is None:
            return None
        source_ts_ns, source_ts_kind = _sample_source_timestamp(sample, self._Gst)
        width, height, fmt = _sample_geometry(sample)
        frame_resource = _sample_frame_resource(
            sample,
            self._Gst,
            width=width,
            height=height,
            pixel_format=fmt,
        )
        if frame_resource is None or not frame_resource.gpu_accessible:
            memory = getattr(frame_resource, "memory", "none")
            raise RuntimeError(
                f"GStreamer resource appsink produced non GPU-accessible memory: {memory}"
            )
        ready_ts_ns = time.monotonic_ns()
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            width=width,
            height=height,
            pixel_format=str(fmt or "").upper(),
            ts_ns=receive_ts_ns,
            capture_wait_ms=(receive_ts_ns - t0) / 1e6,
            image=None,
            frame_resource=frame_resource,
            userspace_process_ms=(ready_ts_ns - receive_ts_ns) / 1e6,
            source_ts_ns=source_ts_ns,
            source_ts_kind=source_ts_kind,
            source_width=self._candidate.source_width,
            source_height=self._candidate.source_height,
            roi_size=self._candidate.roi_size,
            roi_offset_x=self._candidate.roi_offset_x,
            roi_offset_y=self._candidate.roi_offset_y,
        )


def _drain_bus_diagnostics(pipeline: Any, Gst: Any) -> list[str]:
    try:
        bus = pipeline.get_bus()
    except Exception:
        return []
    if bus is None:
        return []
    message_type = getattr(Gst, "MessageType", None)
    if message_type is None:
        return []
    mask = 0
    for name in ("ERROR", "WARNING", "EOS"):
        value = getattr(message_type, name, None)
        if value is not None:
            mask |= value
    if mask == 0:
        return []
    diagnostics: list[str] = []
    while True:
        try:
            message = bus.timed_pop_filtered(0, mask)
        except Exception:
            break
        if message is None:
            break
        diagnostics.append(_format_bus_message(message))
    return diagnostics


def _format_bus_message(message: Any) -> str:
    source = "unknown"
    try:
        source = message.src.get_name()
    except Exception:
        pass
    for parser_name in ("parse_error", "parse_warning"):
        parser = getattr(message, parser_name, None)
        if parser is None:
            continue
        try:
            error, debug = parser()
        except Exception:
            continue
        text = getattr(error, "message", str(error))
        if debug:
            return f"{source}: {text} ({debug})"
        return f"{source}: {text}"
    return f"{source}: {message}"


def _sample_source_timestamp(sample: Any, Gst: Any) -> tuple[int | None, str]:
    try:
        buffer = sample.get_buffer()
    except Exception:
        return None, ""
    none_value = getattr(Gst, "CLOCK_TIME_NONE", None)
    for attr, kind in (("pts", "gstreamer_pts"), ("dts", "gstreamer_dts")):
        try:
            value = getattr(buffer, attr)
        except Exception:
            continue
        if not isinstance(value, int):
            continue
        if value < 0:
            continue
        if none_value is not None and value == none_value:
            continue
        return int(value), kind
    return None, ""


def _sample_frame_resource(
    sample: Any,
    Gst: Any,
    *,
    width: int,
    height: int,
    pixel_format: str,
) -> FrameResource | None:
    try:
        buffer = sample.get_buffer()
    except Exception:
        return None
    features = _sample_caps_features(sample)
    memory_types = _buffer_memory_types(buffer)
    memory = _classify_frame_memory(features, memory_types)
    dmabuf_fd, fd_extraction = _extract_dmabuf_fd_with_diag(buffer)
    gst_buffer_ptr = _gobject_pointer(buffer)
    gst_sample_ptr = _gobject_pointer(sample)
    return FrameResource(
        kind="gstreamer_sample",
        handle=sample,
        memory=memory,
        width=int(width),
        height=int(height),
        pixel_format=str(pixel_format or "").upper(),
        source="appsink",
        dmabuf_fd=dmabuf_fd,
        metadata={
            "caps_features": features,
            "memory_types": memory_types,
            "gst_buffer_pts": _safe_int_attr(buffer, "pts"),
            "gst_buffer_dts": _safe_int_attr(buffer, "dts"),
            "dmabuf_fd": dmabuf_fd,
            "gst_buffer_ptr": gst_buffer_ptr,
            "gst_sample_ptr": gst_sample_ptr,
            "fd_extraction": fd_extraction,
            "memory_kind": memory,
        },
    )


def _sample_caps_features(sample: Any) -> str:
    try:
        caps = sample.get_caps()
        features = caps.get_features(0)
        return str(features.to_string())
    except Exception:
        return ""


def _buffer_memory_types(buffer: Any) -> list[str]:
    try:
        count = int(buffer.n_memory())
    except Exception:
        return []
    result: list[str] = []
    for index in range(max(0, count)):
        try:
            memory = buffer.peek_memory(index)
        except Exception:
            continue
        detected: list[str] = []
        for name in ("NVMM", "DMABuf", "DmaBuf", "CUDA", "GLMemory", "SystemMemory"):
            try:
                if memory.is_type(name):
                    detected.append(name)
            except Exception:
                continue
        result.append(",".join(detected) if detected else type(memory).__name__)
    return result


def _classify_frame_memory(features: str, memory_types: list[str]) -> FrameResourceMemory:
    text = " ".join([features, *memory_types]).lower()
    if "nvmm" in text:
        return "nvmm"
    if "dmabuf" in text or "dmabuf" in text.replace("-", ""):
        return "dmabuf"
    if "cuda" in text or "glmemory" in text:
        return "cuda"
    if "systemmemory" in text or not text.strip():
        return "cpu"
    if memory_types or features:
        return "gstreamer"
    return "unknown"


def _extract_dmabuf_fd(buffer: Any) -> int | None:
    allocators = _gst_allocators_module()
    try:
        count = int(buffer.n_memory())
    except Exception:
        return None
    for index in range(max(0, count)):
        try:
            memory = buffer.peek_memory(index)
        except Exception:
            continue
        fd = _memory_dmabuf_fd(memory, allocators)
        if fd is not None:
            return fd
    return None


def _extract_dmabuf_fd_with_diag(buffer: Any) -> tuple[int | None, dict[str, Any]]:
    """Extract the kernel dmabuf fd from the first sub-memory in the buffer,
    returning both the integer and a structured diagnostic describing how it
    was found (or why it was not).

    On Jetson, appsink fed by `nvvidconv ! video/x-raw(memory:NVMM)` produces
    a GstBuffer whose memory is an NVIDIA NvBuffer allocator memory; that
    memory's `get_fd()` typically returns a `(success: bool, fd: int)` tuple
    rather than a bare int, which the original implementation did not handle.
    """
    allocators = _gst_allocators_module()
    try:
        count = int(buffer.n_memory())
    except Exception:
        return None, {"method": "buffer_count_failed", "buffer_n_memory": 0}
    per_memory: list[dict[str, Any]] = []
    for index in range(max(0, count)):
        try:
            memory = buffer.peek_memory(index)
        except Exception:
            per_memory.append({"index": index, "memory_type": "<peek_failed>"})
            continue
        memory_type = type(memory).__name__
        attrs_seen = [a for a in ("get_fd", "fd", "fileno") if hasattr(memory, a)]
        method = None
        fd: int | None = None
        if allocators is not None:
            try:
                is_dmabuf = bool(allocators.is_dmabuf_memory(memory))
            except Exception:
                is_dmabuf = False
            if is_dmabuf:
                try:
                    raw = allocators.dmabuf_memory_get_fd(memory)
                except Exception:
                    raw = None
                fd = _valid_fd(raw)
                if fd is not None:
                    method = "gst_allocators_dmabuf"
        if fd is None:
            for attr in attrs_seen:
                try:
                    value = getattr(memory, attr)
                except Exception:
                    continue
                if value is None:
                    continue
                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue
                candidates: list[Any] = []
                if isinstance(value, tuple):
                    candidates.extend(value)
                else:
                    candidates.append(value)
                fd_candidate: int | None = None
                for element in candidates:
                    element_fd = _valid_fd(element)
                    if element_fd is None:
                        continue
                    if fd_candidate is None or element_fd > fd_candidate:
                        fd_candidate = element_fd
                if fd_candidate is not None:
                    fd = fd_candidate
                    method = f"tuple_getter_{attr}" if isinstance(value, tuple) else f"attr_{attr}"
                    break
        per_memory.append({
            "index": index,
            "memory_type": memory_type,
            "attrs_seen": attrs_seen,
            "fd": fd,
            "method": method,
        })
        if fd is not None:
            return fd, {
                "method": method,
                "memory_index": index,
                "memory_type": memory_type,
                "attrs_seen": attrs_seen,
                "per_memory": per_memory,
            }
    return None, {
        "method": "none_found",
        "per_memory": per_memory,
        "buffer_n_memory": count,
    }

def _gst_allocators_module() -> Any | None:
    try:
        import gi

        gi.require_version("GstAllocators", "1.0")
        from gi.repository import GstAllocators

        return GstAllocators
    except Exception:
        return None


def _memory_dmabuf_fd(memory: Any, allocators: Any | None) -> int | None:
    if allocators is not None:
        try:
            is_dmabuf = allocators.is_dmabuf_memory(memory)
        except Exception:
            is_dmabuf = False
        if is_dmabuf:
            try:
                fd = allocators.dmabuf_memory_get_fd(memory)
            except Exception:
                fd = None
            result = _valid_fd(fd)
            if result is not None:
                return result
    for attr in ("get_fd", "fd", "fileno"):
        try:
            value = getattr(memory, attr)
        except Exception:
            continue
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, tuple) and value:
            for element in value:
                coerced = _coerce_fd(element)
                if coerced is not None:
                    return coerced
            continue
        coerced = _coerce_fd(value)
        if coerced is not None:
            return coerced
    return None

def _coerce_fd(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        fd = int(value)
    except Exception:
        return None
    return fd if fd >= 3 else None


def _valid_fd(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        fd = int(value)
    except Exception:
        return None
    # Real kernel file descriptors are >= 3 (0/1/2 are stdin/stdout/stderr).
    # This guards against tuple-returning getters like NVIDIA NvBuffer
    # (success, fd) where `success` is `True` and would otherwise coerce
    # to fd=1, which NvBufSurfaceFromFd treats as a usable fd and can
    # subsequently SIGSEGV in the kernel driver.
    return fd if fd >= 3 else None


def _gobject_pointer(obj: Any) -> int | None:
    for attr in ("__gpointer__", "__pointer__", "gpointer"):
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        result = _valid_pointer(value)
        if result is not None:
            return result
    try:
        value = hash(obj)
    except Exception:
        return None
    return _valid_pointer(value)


def _valid_pointer(value: Any) -> int | None:
    try:
        pointer = int(value)
    except Exception:
        return None
    return pointer if pointer > 0 else None


def _safe_int_attr(obj: Any, name: str) -> int | None:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    return int(value) if isinstance(value, int) and value >= 0 else None


def _sample_to_bgr(sample: Any, Gst: Any) -> tuple[Any, int, int, str]:
    import numpy as np

    width, height, fmt = _sample_geometry(sample)
    buffer = sample.get_buffer()
    if _sample_has_gpu_accessible_memory(sample, buffer):
        raise RuntimeError(
            "refusing to map GPU-accessible GStreamer sample into a CPU image; "
            "use GstResourceFrameSource or CaptureLoop for NVMM/DMABUF/CUDA buffers"
        )
    ok, info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("GStreamer sample buffer map failed")
    try:
        raw = np.frombuffer(info.data, dtype=np.uint8)
        if fmt == "BGRx":
            bgrx = raw.reshape((height, width, 4))
            image = np.ascontiguousarray(bgrx[:, :, :3])
        elif fmt == "BGR":
            image = np.ascontiguousarray(raw.reshape((height, width, 3)))
        elif fmt == "NV12":
            image = _nv12_to_bgr(raw, width=width, height=height, np=np)
        else:
            raise RuntimeError(f"unsupported GStreamer sample format: {fmt}")
        return image, width, height, str(fmt or "")
    finally:
        buffer.unmap(info)


def _sample_has_gpu_accessible_memory(sample: Any, buffer: Any) -> bool:
    features = _sample_caps_features(sample).lower()
    memory_types = " ".join(_buffer_memory_types(buffer)).lower()
    text = f"{features} {memory_types}"
    return any(marker in text for marker in ("nvmm", "dmabuf", "cuda", "glmemory"))


def _sample_geometry(sample: Any) -> tuple[int, int, str]:
    caps = sample.get_caps()
    structure = caps.get_structure(0) if caps is not None and caps.get_size() > 0 else None
    if structure is None:
        raise RuntimeError("GStreamer sample has no caps")
    ok_width, width = structure.get_int("width")
    ok_height, height = structure.get_int("height")
    fmt = structure.get_string("format")
    if not ok_width or not ok_height:
        raise RuntimeError("GStreamer sample caps missing width or height")
    return int(width), int(height), str(fmt or "")


def _nv12_to_bgr(raw: Any, *, width: int, height: int, np: Any) -> Any:
    expected = width * height * 3 // 2
    if raw.size < expected:
        raise RuntimeError(
            f"NV12 sample buffer too small: got {raw.size}, expected {expected}"
        )
    y_size = width * height
    y = raw[:y_size].reshape((height, width)).astype(np.int32)
    uv = raw[y_size:expected].reshape((height // 2, width // 2, 2)).astype(np.int32)
    u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)[:height, :width]
    v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)[:height, :width]

    c = np.maximum(y - 16, 0)
    d = u - 128
    e = v - 128
    b = (298 * c + 516 * d + 128) >> 8
    g = (298 * c - 100 * d - 208 * e + 128) >> 8
    r = (298 * c + 409 * e + 128) >> 8
    return np.ascontiguousarray(
        np.stack(
            [
                np.clip(b, 0, 255),
                np.clip(g, 0, 255),
                np.clip(r, 0, 255),
            ],
            axis=2,
        ).astype(np.uint8)
    )
