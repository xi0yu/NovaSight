from __future__ import annotations

import threading
from typing import Generic, TypeVar


T = TypeVar("T")


class LatestFrameQueue(Generic[T]):
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._item: T | None = None
        self._closed = False
        self.dropped = 0
        self.writes = 0
        self.reads = 0

    def put(self, item: T) -> None:
        with self._condition:
            if self._closed:
                return
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self.writes += 1
            self._condition.notify()

    def get(self, timeout: float | None = None) -> T | None:
        with self._condition:
            if self._item is None and not self._closed:
                self._condition.wait(timeout)
            if self._item is None:
                return None
            item = self._item
            self._item = None
            self.reads += 1
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def status(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "queued": self._item is not None,
                "closed": self._closed,
                "dropped": self.dropped,
                "writes": self.writes,
                "reads": self.reads,
                "capacity": 1,
            }
