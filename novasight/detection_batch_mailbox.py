from __future__ import annotations

import threading
import time
from typing import Any

from novasight.contracts import DetectionBatch


class DetectionBatchMailbox:
    """Single-slot latest-only boundary between detection and control."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._batch: DetectionBatch | None = None
        self._published_batches = 0
        self._rejected_batches = 0
        self._overwritten_batches = 0
        self._acquired_batches = 0
        self._last_publish_ts_ns = 0

    def publish(self, batch: DetectionBatch) -> bool:
        if not isinstance(batch, DetectionBatch):
            raise TypeError("DetectionBatchMailbox.publish expects a DetectionBatch")
        with self._condition:
            current = self._batch
            if current is not None and (
                int(batch.generation or 0) <= int(current.generation or 0)
                or int(batch.capture_ts_ns) <= int(current.capture_ts_ns)
            ):
                self._rejected_batches += 1
                return False
            if current is not None:
                self._overwritten_batches += 1
            self._batch = batch
            self._published_batches += 1
            self._last_publish_ts_ns = time.monotonic_ns()
            self._condition.notify_all()
            return True

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
                    self._acquired_batches += 1
                    return self._batch
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def clear(self) -> None:
        with self._condition:
            self._batch = None
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "pending_depth": 1 if self._batch is not None else 0,
                "max_pending_depth": 1,
                "latest_generation": int(self._batch.generation or 0) if self._batch else -1,
                "latest_frame_id": int(self._batch.frame_id) if self._batch else -1,
                "published_batches": self._published_batches,
                "rejected_batches": self._rejected_batches,
                "overwritten_batches": self._overwritten_batches,
                "acquired_batches": self._acquired_batches,
                "last_publish_ts_ns": self._last_publish_ts_ns,
            }

__all__ = ["DetectionBatchMailbox"]
