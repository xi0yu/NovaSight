from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from novasight.capture.source import CapturedFrame

from .failfast import FailFastHandler
from .queue import LatestFrameQueue


@dataclass
class PipelineStats:
    consumed_frames: int = 0
    processed_frames: int = 0
    window_processed_frames: int = 0
    skipped_frames: int = 0
    inference_fps: float = 0.0
    e2e_latency_ms: float = 0.0
    last_frame_id: int = 0
    last_error: str | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    threads: dict[str, bool] = field(default_factory=dict)


class RuntimePipeline:
    CAPTURE_NOT_STARTED_ERROR = "采集未启动，无法运行推理链路。"

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
        self._processed_window_ts_ns: deque[int] = deque()
        self._skipped_window_ts_ns: deque[int] = deque()

    def start(self) -> None:
        if self.running:
            return
        self._require_running_capture()
        self._stop.clear()
        self._last_consumed_frame_id = 0
        self._processed_window_ts_ns.clear()
        self._skipped_window_ts_ns.clear()
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
            skipped = max(0, int(frame.frame_id) - int(self._last_consumed_frame_id) - 1)
            self.stats.consumed_frames += 1
            self._last_consumed_frame_id = frame.frame_id
            self.stats.last_frame_id = self._last_consumed_frame_id
            if skipped:
                now_ns = time.monotonic_ns()
                self._record_skipped(now_ns, skipped)
            if not self._inference_enabled():
                continue
            self.runtime.process_captured_frame(frame)
            self.stats.processed_frames += 1
            done_ns = time.monotonic_ns()
            self._processed_window_ts_ns.append(done_ns)
            self._prune_window(self._processed_window_ts_ns, done_ns)
            self._prune_window(self._skipped_window_ts_ns, done_ns)
            self.stats.window_processed_frames = len(self._processed_window_ts_ns)
            self.stats.inference_fps = self._window_fps(self._processed_window_ts_ns)
            self.stats.e2e_latency_ms = max(0.0, (done_ns - int(frame.ts_ns)) / 1e6)
            self.stats.skipped_frames = len(self._skipped_window_ts_ns)

    def _inference_enabled(self) -> bool:
        config = getattr(self.runtime, "config", None)
        consumers = getattr(config, "consumers", None)
        inference = getattr(config, "inference", None)
        return bool(getattr(consumers, "inference", True)) and bool(
            getattr(inference, "enabled", True)
        )

    def _record_skipped(self, now_ns: int, skipped: int) -> None:
        for _ in range(skipped):
            self._skipped_window_ts_ns.append(now_ns)
        self._prune_window(self._skipped_window_ts_ns, now_ns)
        self.stats.skipped_frames = len(self._skipped_window_ts_ns)

    def _window_fps(self, timestamps_ns: deque[int]) -> float:
        if len(timestamps_ns) < 2:
            return 0.0
        elapsed_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        if elapsed_s <= 0:
            return 0.0
        return (len(timestamps_ns) - 1) / elapsed_s

    def _prune_window(self, timestamps_ns: deque[int], now_ns: int) -> None:
        window_start_ns = now_ns - 1_000_000_000
        while timestamps_ns and timestamps_ns[0] < window_start_ns:
            timestamps_ns.popleft()
