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


class FrameSource(Protocol):
    backend_label: str

    def read(self) -> CapturedFrame | None: ...
    def close(self) -> None: ...


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
            self._stop_pipeline()
            raise RuntimeError("GStreamer pipeline set_state PLAYING failed")
        first = self._pull_frame(timeout_ns=2 * Gst.SECOND)
        if first is None:
            self._stop_pipeline()
            raise RuntimeError("GStreamer appsink first frame timeout")
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
        )


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
