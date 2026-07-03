from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from .pipeline import CaptureCandidate
from .state import CaptureProfile


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    width: int
    height: int
    pixel_format: str
    ts_ns: int
    capture_wait_ms: float
    image: Any
    source_width: int | None = None
    source_height: int | None = None
    roi_size: int | None = None
    roi_offset_x: int = 0
    roi_offset_y: int = 0


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
        self._first_frame: CapturedFrame | None = None
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
        self._first_frame = first

    def opened_and_readable(self) -> bool:
        return self._first_frame is not None

    def read(self) -> CapturedFrame | None:
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
            return frame
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
        t1 = time.monotonic_ns()
        if sample is None:
            return None
        image, width, height = _sample_to_bgr(sample, self._Gst)
        self._frame_id += 1
        return CapturedFrame(
            frame_id=self._frame_id,
            width=width,
            height=height,
            pixel_format="BGR",
            ts_ns=t1,
            capture_wait_ms=(t1 - t0) / 1e6,
            image=image,
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


def _sample_to_bgr(sample: Any, Gst: Any) -> tuple[Any, int, int]:
    import numpy as np

    caps = sample.get_caps()
    structure = caps.get_structure(0) if caps is not None and caps.get_size() > 0 else None
    if structure is None:
        raise RuntimeError("GStreamer sample has no caps")
    ok_width, width = structure.get_int("width")
    ok_height, height = structure.get_int("height")
    fmt = structure.get_string("format")
    if not ok_width or not ok_height:
        raise RuntimeError("GStreamer sample caps missing width or height")
    buffer = sample.get_buffer()
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
        return image, width, height
    finally:
        buffer.unmap(info)


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
