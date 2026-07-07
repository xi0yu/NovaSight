from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from novasight.capture.source import CapturedFrame

from .failfast import FailFastHandler


@dataclass
class PipelineStats:
    consumed_frames: int = 0
    processed_frames: int = 0
    window_processed_frames: int = 0
    skipped_frames: int = 0
    inference_fps: float = 0.0
    e2e_latency_ms: float = 0.0
    queue_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    last_frame_id: int = 0
    last_error: str | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    threads: dict[str, bool] = field(default_factory=dict)


class RuntimePipeline:
    CAPTURE_NOT_STARTED_ERROR = "采集未启动，无法运行推理链路。"
    GPU_BRIDGE_NOT_READY_ERROR = (
        "capture.memory=nvmm 需要可用的 Jetson native bridge，当前未就绪。"
    )

    def __init__(
        self,
        *,
        capture: Any,
        runtime: Any,
        failfast: FailFastHandler | None = None,
    ) -> None:
        self.capture = capture
        self.runtime = runtime
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
        self._require_ready_gpu_preprocessor_if_needed()
        self._stop.clear()
        self._last_consumed_frame_id = 0
        self._processed_window_ts_ns.clear()
        self._skipped_window_ts_ns.clear()
        self.stats.started_at = time.time()
        self.stats.stopped_at = None
        self.runtime.running = True
        self._threads = [
            threading.Thread(
                target=lambda: self.failfast.run("inference_control", self._runtime_loop),
                name="novasight-inference-control",
                daemon=True,
            ),
            threading.Thread(
                target=lambda: self.failfast.run("continuous_control", self._control_loop),
                name="novasight-continuous-control",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
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

    def _require_ready_gpu_preprocessor_if_needed(self) -> None:
        if not self._inference_enabled():
            return
        if self._capture_memory() != "nvmm":
            return
        status = self._gpu_preprocessor_status()
        if status is not None and bool(status.get("available", False)):
            return
        reason = ""
        detail = ""
        if status is None:
            reason = "gpu preprocessor status unavailable"
        else:
            reason = str(status.get("reason") or "gpu preprocessor unavailable")
            detail = str(status.get("detail") or "")
            native_status = status.get("native_status")
            if not detail and isinstance(native_status, dict):
                detail = str(native_status.get("detail") or native_status.get("reason") or "")
        message = f"{self.GPU_BRIDGE_NOT_READY_ERROR} reason={reason}"
        if detail:
            message = f"{message}; detail={detail}"
        self.stats.last_error = message
        raise RuntimeError(message)

    def _capture_memory(self) -> str:
        config = getattr(self.runtime, "config", None)
        capture_config = getattr(config, "capture", None)
        memory = getattr(capture_config, "memory", None)
        if memory is None:
            capture_config = getattr(self.capture, "config", None)
            memory = getattr(capture_config, "memory", None)
        return str(memory or "cpu").lower()

    def _gpu_preprocessor_status(self) -> dict[str, Any] | None:
        runtime_status = getattr(self.runtime, "status", None)
        if callable(runtime_status):
            try:
                status = runtime_status()
            except Exception as exc:
                return {
                    "available": False,
                    "reason": "runtime status failed",
                    "detail": str(exc),
                }
            if isinstance(status, dict):
                bridge = status.get("gpu_preprocessor")
                if isinstance(bridge, dict):
                    return bridge
        preprocessor = getattr(self.runtime, "_gpu_preprocessor", None)
        status_fn = getattr(preprocessor, "status", None)
        if callable(status_fn):
            try:
                status = status_fn()
            except Exception as exc:
                return {
                    "available": False,
                    "reason": "gpu preprocessor status failed",
                    "detail": str(exc),
                }
            if isinstance(status, dict):
                return status
        return None

    def _capture_is_still_available(self) -> bool:
        state = getattr(self.capture, "state", None)
        if getattr(state, "available", False) is not True:
            return False
        session = getattr(self.capture, "session", None)
        if session is not None and getattr(session, "running", False) is not True:
            return False
        if getattr(self.capture, "source", None) is None:
            return False
        return True

    def _capture_unavailable_reason(self) -> str:
        state = getattr(self.capture, "state", None)
        reason = str(getattr(state, "last_error", "") or "").strip()
        if reason:
            return reason
        session = getattr(self.capture, "session", None)
        if session is not None and getattr(session, "running", False) is not True:
            return "capture session stopped"
        if getattr(self.capture, "source", None) is None:
            return "capture source unavailable"
        return "capture unavailable"

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
                if not self._capture_is_still_available():
                    self.stats.last_error = self._capture_unavailable_reason()
                    self.runtime.running = False
                    self._stop.set()
                    break
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
            process_start_ns = time.monotonic_ns()
            self.runtime.process_captured_frame(frame)
            self.stats.processed_frames += 1
            done_ns = time.monotonic_ns()
            self._processed_window_ts_ns.append(done_ns)
            self._prune_window(self._processed_window_ts_ns, done_ns)
            self._prune_window(self._skipped_window_ts_ns, done_ns)
            self.stats.window_processed_frames = len(self._processed_window_ts_ns)
            self.stats.inference_fps = self._window_fps(self._processed_window_ts_ns)
            self.stats.queue_latency_ms = max(0.0, (process_start_ns - int(frame.capture_ts_ns)) / 1e6)
            self.stats.inference_latency_ms = max(0.0, (done_ns - process_start_ns) / 1e6)
            self.stats.e2e_latency_ms = max(0.0, (done_ns - int(frame.capture_ts_ns)) / 1e6)
            self.stats.skipped_frames = len(self._skipped_window_ts_ns)

    def _control_loop(self) -> None:
        process_control_tick = getattr(self.runtime, "process_control_tick", None)
        if not callable(process_control_tick):
            return
        while not self._stop.is_set():
            if not self._continuous_control_enabled():
                time.sleep(0.05)
                continue
            process_control_tick()
            time.sleep(self._control_interval_s())

    def _inference_enabled(self) -> bool:
        config = getattr(self.runtime, "config", None)
        consumers = getattr(config, "consumers", None)
        inference = getattr(config, "inference", None)
        return bool(getattr(consumers, "inference", True)) and bool(
            getattr(inference, "enabled", True)
        )

    def _continuous_control_enabled(self) -> bool:
        config = getattr(self.runtime, "config", None)
        control = getattr(config, "control", None)
        return str(getattr(control, "strategy", "")) == "experimental_angle_pid"

    def _control_interval_s(self) -> float:
        config = getattr(self.runtime, "config", None)
        control = getattr(config, "control", None)
        hz = max(1.0, float(getattr(control, "experimental_angle_control_hz", 60.0)))
        return max(0.001, min(0.05, 1.0 / hz))

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
