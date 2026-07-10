from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace

from novasight.runtime.latest_frame import FrameHandle, LatestFrameBroker

from .source import CapturedFrame, FrameSource
from .state import CaptureProfile, CaptureRuntimeState

logger = logging.getLogger("novasight.capture.session")


class CaptureSession:
    def __init__(
        self,
        *,
        source_factory: Callable[[CaptureProfile], FrameSource],
        empty_read_sleep_s: float = 0.001,
        stop_timeout_s: float = 1.0,
    ) -> None:
        self.source_factory = source_factory
        self.empty_read_sleep_s = empty_read_sleep_s
        self.stop_timeout_s = stop_timeout_s
        self.state = CaptureRuntimeState()
        self._source: FrameSource | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._lifecycle_lock = threading.RLock()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._latest_frame: CapturedFrame | None = None
        self._last_frame_ts_ns: int | None = None
        self._capture_window_ts_ns: deque[int] = deque()
        self._drop_window_ts_ns: deque[int] = deque()
        self.latest_frame_broker = LatestFrameBroker()
        self._broker_generation = 0

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def source(self) -> FrameSource | None:
        with self._lock:
            return self._source

    def start(self, profile: CaptureProfile) -> CaptureRuntimeState:
        with self._lifecycle_lock:
            if self._thread is not None or self.source is not None:
                stopped = self._stop_current_locked("restarting capture")
                thread = self._thread
                if thread is not None and thread.is_alive():
                    raise RuntimeError(stopped.last_error or "capture thread did not stop")
            try:
                source = self.source_factory(profile)
            except Exception as exc:
                with self._condition:
                    self.state = replace(
                        self.state,
                        available=False,
                        device=profile.device,
                        profile=profile,
                        backend=None,
                        last_error=str(exc),
                    )
                    self._condition.notify_all()
                raise
            stop_event = threading.Event()
            with self._condition:
                self._source = source
                self._latest_frame = None
                self._last_frame_ts_ns = None
                self._broker_generation = 0
                self.latest_frame_broker.clear()
                self._capture_window_ts_ns.clear()
                self._drop_window_ts_ns.clear()
                self._stop_event = stop_event
                self.state = CaptureRuntimeState(
                    available=True,
                    device=profile.device,
                    profile=profile,
                    backend=source.backend_label,
                    last_error=None,
                )
                self._thread = threading.Thread(
                    target=self._run_loop,
                    args=(source, stop_event),
                    name="novasight-capture-session",
                    daemon=True,
                )
                self._thread.start()
                self._condition.notify_all()
                return self.state

    def reconfigure(self, profile: CaptureProfile) -> CaptureRuntimeState:
        return self.start(profile)

    def stop(self, reason: str | None = None) -> CaptureRuntimeState:
        with self._lifecycle_lock:
            return self._stop_current_locked(reason)

    def _stop_current_locked(self, reason: str | None = None) -> CaptureRuntimeState:
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(self.stop_timeout_s)
            if thread.is_alive():
                timeout_text = f"{self.stop_timeout_s:.3f}".rstrip("0").rstrip(".")
                if "." not in timeout_text:
                    timeout_text = f"{timeout_text}.0"
                reason_text = (
                    f"capture thread did not stop within {timeout_text}s"
                )
                with self._condition:
                    self.state = replace(
                        self.state,
                        available=False,
                        last_error=reason_text,
                    )
                    self._condition.notify_all()
                    return self.state
        with self._condition:
            close_error = self._close_source_locked()
            self._thread = None
            self._stop_event = None
            self.state = replace(
                self.state,
                available=False,
                last_error=close_error or reason or self.state.last_error,
            )
            self._latest_frame = None
            self._last_frame_ts_ns = None
            self._broker_generation = 0
            self.latest_frame_broker.clear()
            self._capture_window_ts_ns.clear()
            self._condition.notify_all()
            return self.state

    def latest_frame(
        self,
        after_frame_id: int | None = None,
        timeout_s: float = 0.0,
    ) -> CapturedFrame | None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self._condition:
            while True:
                frame = self._latest_frame
                if frame is not None and (
                    after_frame_id is None or frame.frame_id > after_frame_id
                ):
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def _run_loop(
        self,
        source: FrameSource,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                frame = source.read()
            except Exception as exc:
                logger.warning("capture session read failed: %s", exc)
                self._mark_unavailable(
                    f"capture read failed: {exc}",
                    source=source,
                    stop_event=stop_event,
                )
                return
            if frame is None:
                with self._lock:
                    if self._source is source:
                        self.state.frames_dropped += 1
                        now_ns = time.monotonic_ns()
                        self._drop_window_ts_ns.append(now_ns)
                        self._prune_window(self._drop_window_ts_ns, now_ns)
                        self.state.statistics.dropped_counter = len(self._drop_window_ts_ns)
                if self.empty_read_sleep_s > 0:
                    time.sleep(self.empty_read_sleep_s)
                continue
            self._publish_frame(frame, source=source, stop_event=stop_event)

    def _publish_frame(
        self,
        frame: CapturedFrame,
        *,
        source: FrameSource,
        stop_event: threading.Event,
    ) -> None:
        with self._condition:
            if stop_event.is_set() or self._source is not source:
                return
            timestamp_error = self._frame_timestamp_error(frame)
            if timestamp_error:
                self._mark_unavailable(
                    timestamp_error,
                    source=source,
                    stop_event=stop_event,
                )
                stop_event.set()
                return
            self.state.available = True
            self.state.capture_wait_ms = frame.capture_wait_ms
            if self._last_frame_ts_ns is not None:
                self.state.frame_period_ms = (frame.capture_ts_ns - self._last_frame_ts_ns) / 1e6
            self._capture_window_ts_ns.append(frame.capture_ts_ns)
            self._prune_window(self._capture_window_ts_ns, frame.capture_ts_ns)
            self._prune_window(self._drop_window_ts_ns, frame.capture_ts_ns)
            self.state.statistics.capture_counter = len(self._capture_window_ts_ns)
            self.state.statistics.dropped_counter = len(self._drop_window_ts_ns)
            self.state.fps_capture = self._window_fps(self._capture_window_ts_ns)
            self.state.statistics.capture_fps = self.state.fps_capture
            self._last_frame_ts_ns = frame.capture_ts_ns
            self._broker_generation += 1
            frame = replace(frame, generation=self._broker_generation)
            self._latest_frame = frame
            self.latest_frame_broker.publish(
                _frame_handle_from_capture_frame(
                    frame,
                    generation=self._broker_generation,
                )
            )
            broker_status = self.latest_frame_broker.status()
            self.state.statistics.published_frames = int(broker_status.get("published_frames", 0))
            self.state.statistics.overwritten_frames = int(broker_status.get("overwritten_frames", 0))
            self.state.statistics.acquired_frames = int(broker_status.get("acquired_frames", 0))
            self.state.statistics.latest_frame_age_ms = max(
                0.0,
                (time.monotonic_ns() - int(frame.capture_ts_ns)) / 1e6,
            )
            self.state.statistics.appsink_caps = str(getattr(frame, "caps_string", "") or "")
            self.state.statistics.actual_pipeline_string = str(
                getattr(frame, "actual_pipeline_string", "") or ""
            )
            self.state.statistics.content_validation_status = "not_integrated"
            self.state.statistics.content_validation_reason = (
                "NVMM zero-copy frame has no GPU luma/variance probe"
                if frame.image is None
                else "CPU frame content metrics are not enabled"
            )
            self.state.last_error = None
            self._condition.notify_all()

    def _frame_timestamp_error(self, frame: CapturedFrame) -> str:
        try:
            capture_ts_ns = int(frame.capture_ts_ns)
        except Exception:
            return "capture frame timestamp invalid: capture_ts_ns is not an integer"
        if capture_ts_ns <= 0:
            return (
                "capture frame timestamp invalid: "
                f"capture_ts_ns must be positive, got {capture_ts_ns}"
            )
        if self._last_frame_ts_ns is not None and capture_ts_ns < self._last_frame_ts_ns:
            return (
                "capture frame timestamp went backwards: "
                f"previous={self._last_frame_ts_ns} current={capture_ts_ns}"
            )
        return ""

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

    def _mark_unavailable(
        self,
        reason: str,
        *,
        source: FrameSource,
        stop_event: threading.Event,
    ) -> None:
        with self._condition:
            if stop_event.is_set() or self._source is not source:
                return
            close_error = self._close_source_locked()
            self._thread = None
            self._stop_event = None
            self.state = replace(
                self.state,
                available=False,
                last_error=f"{reason}; {close_error}" if close_error else reason,
            )
            self._latest_frame = None
            self._last_frame_ts_ns = None
            self._broker_generation = 0
            self.latest_frame_broker.clear()
            self._capture_window_ts_ns.clear()
            self._drop_window_ts_ns.clear()
            self._condition.notify_all()

    def _close_source_locked(self) -> str:
        source = self._source
        self._source = None
        if source is None:
            return ""
        try:
            source.close()
        except Exception as exc:
            return f"capture close failed: {exc}"
        return ""


def _frame_handle_from_capture_frame(frame: CapturedFrame, *, generation: int) -> FrameHandle:
    frame_resource = getattr(frame, "frame_resource", None)
    resource = getattr(frame_resource, "handle", None)
    if resource is None:
        resource = getattr(frame, "image", None)
    if resource is None:
        resource = frame
    running_time_ns = getattr(frame, "source_ts_ns", None)
    return FrameHandle(
        generation=int(generation),
        frame_id=int(frame.frame_id),
        source_sequence=int(frame.frame_id),
        capture_ts_ns=int(frame.capture_ts_ns),
        clock_domain="monotonic",
        pipeline_running_time_ns=int(running_time_ns) if running_time_ns is not None else None,
        width=int(frame.width),
        height=int(frame.height),
        format=str(frame.pixel_format),
        resource=resource,
        metadata={
            "captured_frame": frame,
            "resource_memory": str(getattr(frame, "resource_memory", "cpu")),
            "source_ts_ns": getattr(frame, "source_ts_ns", None),
            "source_ts_kind": str(getattr(frame, "source_ts_kind", "") or ""),
            "capture_ts_source": str(getattr(frame, "capture_ts_source", "monotonic")),
            "content_validation_status": "not_integrated",
            "content_validation_reason": (
                "NVMM zero-copy frame has no GPU luma/variance probe"
                if frame.image is None
                else "CPU frame content metrics are not enabled"
            ),
        },
    )
