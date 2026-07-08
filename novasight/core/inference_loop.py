from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from novasight.contracts import DetectionBatch
from novasight.pipelines.probes import DetectionSlot


DetectionBatchConsumer = Callable[[DetectionBatch], Any]


@dataclass
class InferenceLoopStats:
    consumed_batches: int = 0
    skipped_batches: int = 0
    last_frame_id: int = -1
    last_error: str = ""
    started_at_ns: int = 0
    stopped_at_ns: int = 0


class InferenceLoop(threading.Thread):
    """Consume DetectionSlot updates without doing work inside the DeepStream probe."""

    def __init__(
        self,
        *,
        slot: DetectionSlot,
        consumer: DetectionBatchConsumer,
        poll_timeout_s: float = 0.1,
        name: str = "novasight-inference-loop",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.slot = slot
        self.consumer = consumer
        self.poll_timeout_s = max(0.0, float(poll_timeout_s))
        self.stats = InferenceLoopStats()
        self._stop_event = threading.Event()
        self._last_slot_version = 0

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=max(0.0, float(timeout_s)))

    def run(self) -> None:
        self.stats.started_at_ns = time.monotonic_ns()
        self.stats.stopped_at_ns = 0
        try:
            while not self._stop_event.is_set():
                batch = self.slot.get(
                    after_version=self._last_slot_version,
                    timeout_s=self.poll_timeout_s,
                )
                if batch is None:
                    continue
                current_version = self.slot.version
                skipped = max(0, current_version - self._last_slot_version - 1)
                self._last_slot_version = current_version
                self.stats.skipped_batches += skipped
                self.stats.consumed_batches += 1
                self.stats.last_frame_id = int(batch.frame_id)
                self.consumer(batch)
        except Exception as exc:
            self.stats.last_error = str(exc)
        finally:
            self.stats.stopped_at_ns = time.monotonic_ns()


__all__ = [
    "DetectionBatchConsumer",
    "InferenceLoop",
    "InferenceLoopStats",
]
