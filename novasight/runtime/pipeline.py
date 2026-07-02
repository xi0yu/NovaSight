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
    consumed_frames: int = 0
    processed_frames: int = 0
    last_frame_id: int = 0
    last_error: str | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    threads: dict[str, bool] = field(default_factory=dict)


class RuntimePipeline:
    CAPTURE_NOT_STARTED_ERROR = "采集未启动，无法启动推理控制。"

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
        self._last_consumed_frame_id = 0

    def start(self) -> None:
        if self.running:
            return
        self._require_running_capture()
        self._stop.clear()
        self._last_consumed_frame_id = 0
        self.frame_queue = LatestFrameQueue[CapturedFrame]()
        self.stats.started_at = time.time()
        self.stats.stopped_at = None
        self.runtime.running = True
        self._threads = [
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

    def _require_running_capture(self) -> None:
        state = getattr(self.capture, "state", None)
        session = getattr(self.capture, "session", None)
        if (
            getattr(self.capture, "source", None) is None
            or getattr(state, "available", False) is not True
            or (session is not None and getattr(session, "running", False) is not True)
        ):
            raise RuntimeError(self.CAPTURE_NOT_STARTED_ERROR)

    def _runtime_loop(self) -> None:
        wait_frame = getattr(self.capture, "wait_preview_frame", None)
        if not callable(wait_frame):
            wait_frame = getattr(self.capture, "latest_frame")
        while not self._stop.is_set():
            frame = wait_frame(
                after_frame_id=self._last_consumed_frame_id,
                timeout_s=0.1,
            )
            if frame is None:
                continue
            self.runtime.process_captured_frame(frame)
            self.stats.consumed_frames += 1
            self.stats.processed_frames += 1
            self._last_consumed_frame_id = frame.frame_id
            self.stats.last_frame_id = self._last_consumed_frame_id
