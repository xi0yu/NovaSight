from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class FrameHandle:
    generation: int
    frame_id: int
    source_sequence: int
    capture_ts_ns: int
    clock_domain: str
    pipeline_running_time_ns: int | None
    width: int
    height: int
    format: str
    resource: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    release_callback: Callable[[Any], None] | None = None

    def __post_init__(self) -> None:
        if int(self.generation) < 0:
            raise ValueError("FrameHandle.generation must be >= 0")
        if int(self.frame_id) < 0:
            raise ValueError("FrameHandle.frame_id must be >= 0")
        if int(self.source_sequence) < 0:
            raise ValueError("FrameHandle.source_sequence must be >= 0")
        if int(self.capture_ts_ns) <= 0:
            raise ValueError("FrameHandle.capture_ts_ns must be positive")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("FrameHandle width/height must be positive")
        if not str(self.clock_domain).strip():
            raise ValueError("FrameHandle.clock_domain must be non-empty")
        if not str(self.format).strip():
            raise ValueError("FrameHandle.format must be non-empty")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "_released", False)

    def release(self) -> None:
        if getattr(self, "_released", False):
            return
        object.__setattr__(self, "_released", True)
        callback = self.release_callback
        if callback is not None:
            callback(self.resource)


class LatestFrameBroker:
    """Single-slot latest-frame mailbox for runtime-owned inference scheduling."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: FrameHandle | None = None
        self._published_generation = -1
        self._published_frame_id = -1
        self._published_capture_ts_ns = 0
        self._published_width = 0
        self._published_height = 0
        self._published_format = ""
        self._published_resource_memory = ""
        self._published_capture_ts_source = ""
        self._published_content_validation_status = ""
        self._published_content_validation_reason = ""
        self._acquired_generation = -1
        self._acquired_frame_id = -1
        self._published_frames = 0
        self._overwritten_frames = 0
        self._stale_published_frames = 0
        self._acquired_frames = 0
        self._last_publish_ts_ns = 0
        self._last_acquire_ts_ns = 0

    def publish(self, frame: FrameHandle) -> None:
        if not isinstance(frame, FrameHandle):
            raise TypeError("LatestFrameBroker.publish expects a FrameHandle")
        old: FrameHandle | None = None
        stale: FrameHandle | None = None
        with self._condition:
            if (
                int(frame.generation) <= self._published_generation
                or int(frame.frame_id) <= self._published_frame_id
            ):
                self._stale_published_frames += 1
                stale = frame
            elif self._pending is not None and int(self._pending.generation) != int(frame.generation):
                self._overwritten_frames += 1
                old = self._pending
                self._pending = frame
                self._published_generation = int(frame.generation)
                self._published_frame_id = int(frame.frame_id)
                self._record_published_frame(frame)
                self._published_frames += 1
                self._last_publish_ts_ns = time.monotonic_ns()
                self._condition.notify_all()
            else:
                self._pending = frame
                self._published_generation = int(frame.generation)
                self._published_frame_id = int(frame.frame_id)
                self._record_published_frame(frame)
                self._published_frames += 1
                self._last_publish_ts_ns = time.monotonic_ns()
                self._condition.notify_all()
        if old is not None:
            old.release()
        if stale is not None:
            stale.release()

    def acquire_latest(
        self,
        *,
        after_generation: int,
        timeout_s: float,
    ) -> FrameHandle | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        after = int(after_generation)
        with self._condition:
            while True:
                if self._pending is not None and int(self._pending.generation) > after:
                    frame = self._pending
                    self._pending = None
                    self._acquired_generation = int(frame.generation)
                    self._acquired_frame_id = int(frame.frame_id)
                    self._acquired_frames += 1
                    self._last_acquire_ts_ns = time.monotonic_ns()
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def clear(self) -> None:
        with self._condition:
            old = self._pending
            self._pending = None
            self._published_generation = -1
            self._published_frame_id = -1
            self._published_capture_ts_ns = 0
            self._published_width = 0
            self._published_height = 0
            self._published_format = ""
            self._published_resource_memory = ""
            self._published_capture_ts_source = ""
            self._published_content_validation_status = ""
            self._published_content_validation_reason = ""
            self._acquired_generation = -1
            self._acquired_frame_id = -1
            self._published_frames = 0
            self._overwritten_frames = 0
            self._stale_published_frames = 0
            self._acquired_frames = 0
            self._last_publish_ts_ns = 0
            self._last_acquire_ts_ns = 0
            self._condition.notify_all()
        if old is not None:
            old.release()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "pending_depth": 1 if self._pending is not None else 0,
                "max_pending_depth": 1,
                "published_generation": self._published_generation,
                "published_frame_id": self._published_frame_id,
                "published_capture_ts_ns": self._published_capture_ts_ns,
                "published_frame_age_ms": (
                    max(0.0, (time.monotonic_ns() - self._published_capture_ts_ns) / 1e6)
                    if self._published_capture_ts_ns > 0
                    else None
                ),
                "published_width": self._published_width,
                "published_height": self._published_height,
                "published_format": self._published_format,
                "published_resource_memory": self._published_resource_memory,
                "published_capture_ts_source": self._published_capture_ts_source,
                "published_content_validation_status": self._published_content_validation_status,
                "published_content_validation_reason": self._published_content_validation_reason,
                "acquired_generation": self._acquired_generation,
                "acquired_frame_id": self._acquired_frame_id,
                "published_frames": self._published_frames,
                "overwritten_frames": self._overwritten_frames,
                "latest_overwrite_count": self._overwritten_frames,
                "busy_drop_count": self._overwritten_frames,
                "stale_published_frames": self._stale_published_frames,
                "acquired_frames": self._acquired_frames,
                "last_publish_ts_ns": self._last_publish_ts_ns,
                "last_acquire_ts_ns": self._last_acquire_ts_ns,
            }

    def _record_published_frame(self, frame: FrameHandle) -> None:
        self._published_capture_ts_ns = int(frame.capture_ts_ns)
        self._published_width = int(frame.width)
        self._published_height = int(frame.height)
        self._published_format = str(frame.format)
        self._published_resource_memory = str(frame.metadata.get("resource_memory") or "")
        self._published_capture_ts_source = str(frame.metadata.get("capture_ts_source") or "")
        self._published_content_validation_status = str(
            frame.metadata.get("content_validation_status") or ""
        )
        self._published_content_validation_reason = str(
            frame.metadata.get("content_validation_reason") or ""
        )


LatestFrameExchange = LatestFrameBroker


__all__ = ["FrameHandle", "LatestFrameBroker", "LatestFrameExchange"]
