from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from novasight.capture.source import CapturedFrame

from .failfast import FailFastHandler
from .queue import LatestFrameQueue


@dataclass
class PipelineStats:
    capture_frames: int = 0
    processed_frames: int = 0
    last_frame_id: int = 0
    last_error: str | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    threads: dict[str, bool] = field(default_factory=dict)


class RuntimePipeline:
    def __init__(
        self,
        *,
        capture: Any,
        runtime: Any,
        frame_queue: LatestFrameQueue[CapturedFrame] | None = None,
        failfast: FailFastHandler | None = None,
    ) -> None:
        self.capture = capture
        self.runtime = runtime
        self.frame_queue = frame_queue or LatestFrameQueue[CapturedFrame]()
        self.failfast = failfast or FailFastHandler(
            on_fatal=getattr(runtime, "record_fatal_error", None)
        )
        self.stats = PipelineStats()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self.running:
            return
        if self.capture.source is None:
            state = self.capture.configure(self.capture.config.device)
            if not state.available:
                raise RuntimeError(state.last_error or "capture source failed to open")
        self._stop.clear()
        self.frame_queue = LatestFrameQueue[CapturedFrame]()
        self.stats.started_at = time.time()
        self.stats.stopped_at = None
        self.runtime.running = True
        self._threads = [
            threading.Thread(
                target=lambda: self.failfast.run("capture", self._capture_loop),
                name="novasight-capture",
                daemon=True,
            ),
            threading.Thread(
                target=lambda: self.failfast.run("inference_control", self._runtime_loop),
                name="novasight-inference-control",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.frame_queue.close()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self.runtime.running = False
        self.stats.stopped_at = time.time()

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def status(self) -> dict[str, Any]:
        self.stats.threads = {thread.name: thread.is_alive() for thread in self._threads}
        return {
            **self.stats.__dict__,
            "queue": self.frame_queue.status(),
            "running": self.running,
        }

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.capture.read_frame()
            if frame is None:
                continue
            self.stats.capture_frames += 1
            self.stats.last_frame_id = frame.frame_id
            self.frame_queue.put(frame)

    def _runtime_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.frame_queue.get(timeout=0.1)
            if frame is None:
                continue
            self.runtime.process_captured_frame(frame)
            self.stats.processed_frames += 1
            self.stats.last_frame_id = frame.frame_id
