from __future__ import annotations

import threading
import time
from typing import Any

from novasight.contracts import DetectionBatch


class DetectionBatchMailbox:
    """Latest-only DetectionBatch slot for the control mainline."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._batch: DetectionBatch | None = None
        self._latest_generation = -1
        self._latest_frame_id = -1
        self._latest_capture_ts_ns = 0
        self._published_batches = 0
        self._stale_published_batches = 0
        self._overwritten_batches = 0
        self._last_publish_ts_ns = 0

    def publish(self, batch: DetectionBatch) -> None:
        if not isinstance(batch, DetectionBatch):
            raise TypeError("DetectionBatchMailbox.publish expects a DetectionBatch")
        with self._condition:
            generation = int(batch.generation or 0)
            frame_id = int(batch.frame_id)
            capture_ts_ns = int(batch.capture_ts_ns)
            if (
                bool(batch.is_stale)
                or generation <= self._latest_generation
                or frame_id <= self._latest_frame_id
                or capture_ts_ns <= self._latest_capture_ts_ns
            ):
                self._stale_published_batches += 1
                self._last_publish_ts_ns = time.monotonic_ns()
                self._condition.notify_all()
                return
            if self._batch is not None and int(self._batch.generation or 0) != generation:
                self._overwritten_batches += 1
            self._batch = batch
            self._latest_generation = generation
            self._latest_frame_id = frame_id
            self._latest_capture_ts_ns = capture_ts_ns
            self._published_batches += 1
            self._last_publish_ts_ns = time.monotonic_ns()
            self._condition.notify_all()

    def acquire_latest(
        self,
        *,
        after_generation: int,
        timeout_s: float,
    ) -> DetectionBatch | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        after = int(after_generation)
        with self._condition:
            while True:
                if self._batch is not None and int(self._batch.generation or 0) > after:
                    return self._batch
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def clear(self) -> None:
        with self._condition:
            self._batch = None
            self._latest_generation = -1
            self._latest_frame_id = -1
            self._latest_capture_ts_ns = 0
            self._published_batches = 0
            self._stale_published_batches = 0
            self._overwritten_batches = 0
            self._last_publish_ts_ns = 0
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "pending_depth": 1 if self._batch is not None else 0,
                "max_pending_depth": 1,
                "latest_generation": self._latest_generation,
                "latest_frame_id": self._latest_frame_id,
                "latest_capture_ts_ns": self._latest_capture_ts_ns,
                "published_batches": self._published_batches,
                "stale_published_batches": self._stale_published_batches,
                "overwritten_batches": self._overwritten_batches,
                "last_publish_ts_ns": self._last_publish_ts_ns,
            }


__all__ = ["DetectionBatchMailbox"]
