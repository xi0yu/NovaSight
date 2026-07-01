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
