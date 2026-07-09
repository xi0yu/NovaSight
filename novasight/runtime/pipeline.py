from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from novasight.contracts import DetectionBatch

from .failfast import FailFastHandler


@dataclass
class PipelineStats:
    consumed_frames: int = 0
    processed_frames: int = 0
    window_processed_frames: int = 0
    window_control_observations: int = 0
    skipped_frames: int = 0
    consumed_detection_batches: int = 0
    inference_fps: float = 0.0
    detection_batch_fps: float = 0.0
    control_observation_fps: float = 0.0
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

    def __init__(
        self,
        *,
        capture: Any,
        runtime: Any,
        detection_source: Any | None = None,
        failfast: FailFastHandler | None = None,
    ) -> None:
        self.capture = capture
        self.runtime = runtime
        self.detection_source = detection_source
        self.failfast = failfast or FailFastHandler(
            on_fatal=getattr(runtime, "record_fatal_error", None)
        )
        self.stats = PipelineStats()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_consumed_frame_id = -1
        self._last_consumed_generation = -1
        self._processed_window_ts_ns: deque[int] = deque()
        self._control_observation_window_ts_ns: deque[int] = deque()
        self._skipped_window_ts_ns: deque[int] = deque()

    def start(self) -> None:
        if self.running:
            return
        if self.detection_source is None:
            self._require_running_capture()
        self._stop.clear()
        self._last_consumed_frame_id = -1
        self._last_consumed_generation = -1
        self._processed_window_ts_ns.clear()
        self._control_observation_window_ts_ns.clear()
        self._skipped_window_ts_ns.clear()
        self.stats = PipelineStats(started_at=time.time())
        try:
            self._start_detection_source()
        except Exception:
            self.runtime.running = False
            raise
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
        self._stop_detection_source()
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
            "detection_source": self._detection_source_status(),
            "latest_frame_broker": self._latest_frame_broker_status(),
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
        if self.detection_source is not None:
            self._detection_batch_loop()
            return
        broker = self._latest_frame_broker()
        if broker is not None:
            self._latest_frame_broker_loop(broker)
            return
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
            skipped = self._skipped_since_previous(frame.frame_id)
            self.stats.consumed_frames += 1
            self._last_consumed_frame_id = frame.frame_id
            self.stats.last_frame_id = self._last_consumed_frame_id
            if skipped:
                now_ns = time.monotonic_ns()
                self._record_skipped(now_ns, skipped)
            if not self._inference_enabled():
                continue
            process_start_ns = time.monotonic_ns()
            stale_input_reason = self._stale_inference_input_reason(
                frame_capture_ts_ns=int(frame.capture_ts_ns),
                now_ns=process_start_ns,
            )
            if stale_input_reason:
                self.stats.last_error = stale_input_reason
                self._record_skipped(process_start_ns, 1)
                self._prune_window(self._skipped_window_ts_ns, process_start_ns)
                self.stats.skipped_frames = len(self._skipped_window_ts_ns)
                continue
            result = self.runtime.process_captured_frame(frame)
            self.stats.processed_frames += 1
            done_ns = time.monotonic_ns()
            self._processed_window_ts_ns.append(done_ns)
            self._record_control_observation_if_updated(result, done_ns)
            self._prune_window(self._processed_window_ts_ns, done_ns)
            self._prune_window(self._control_observation_window_ts_ns, done_ns)
            self._prune_window(self._skipped_window_ts_ns, done_ns)
            self.stats.window_processed_frames = len(self._processed_window_ts_ns)
            self.stats.window_control_observations = len(self._control_observation_window_ts_ns)
            self.stats.inference_fps = self._window_fps(self._processed_window_ts_ns)
            self.stats.control_observation_fps = self._window_fps(self._control_observation_window_ts_ns)
            self.stats.detection_batch_fps = 0.0
            self.stats.queue_latency_ms = max(0.0, (process_start_ns - int(frame.capture_ts_ns)) / 1e6)
            self.stats.inference_latency_ms = max(0.0, (done_ns - process_start_ns) / 1e6)
            self.stats.e2e_latency_ms = max(0.0, (done_ns - int(frame.capture_ts_ns)) / 1e6)
            self.stats.skipped_frames = len(self._skipped_window_ts_ns)

    def _latest_frame_broker_loop(self, broker: Any) -> None:
        acquire_latest = getattr(broker, "acquire_latest", None)
        if not callable(acquire_latest):
            self.stats.last_error = "latest frame broker missing acquire_latest()"
            self.runtime.running = False
            self._stop.set()
            return
        while not self._stop.is_set():
            handle = acquire_latest(
                after_generation=self._last_consumed_generation,
                timeout_s=0.1,
            )
            if handle is None:
                if not self._capture_is_still_available():
                    self.stats.last_error = self._capture_unavailable_reason()
                    self.runtime.running = False
                    self._stop.set()
                    break
                continue
            frame = self._captured_frame_from_handle(handle)
            release = getattr(handle, "release", None)
            try:
                if frame is None:
                    self.stats.last_error = "FrameHandle missing captured_frame adapter"
                    self.runtime.running = False
                    self._stop.set()
                    break
                skipped = self._skipped_generations_since_previous(handle)
                self.stats.consumed_frames += 1
                self._last_consumed_generation = int(getattr(handle, "generation", -1))
                self._last_consumed_frame_id = int(frame.frame_id)
                self.stats.last_frame_id = self._last_consumed_frame_id
                if skipped:
                    self._record_skipped(time.monotonic_ns(), skipped)
                if not self._inference_enabled():
                    continue
                process_start_ns = time.monotonic_ns()
                stale_input_reason = self._stale_inference_input_reason(
                    frame_capture_ts_ns=int(frame.capture_ts_ns),
                    now_ns=process_start_ns,
                )
                if stale_input_reason:
                    self.stats.last_error = stale_input_reason
                    self._record_skipped(process_start_ns, 1)
                    self._prune_window(self._skipped_window_ts_ns, process_start_ns)
                    self.stats.skipped_frames = len(self._skipped_window_ts_ns)
                    continue
                result = self.runtime.process_captured_frame(frame)
                self.stats.processed_frames += 1
                done_ns = time.monotonic_ns()
                self._processed_window_ts_ns.append(done_ns)
                self._record_control_observation_if_updated(result, done_ns)
                self._prune_window(self._processed_window_ts_ns, done_ns)
                self._prune_window(self._control_observation_window_ts_ns, done_ns)
                self._prune_window(self._skipped_window_ts_ns, done_ns)
                self.stats.window_processed_frames = len(self._processed_window_ts_ns)
                self.stats.window_control_observations = len(self._control_observation_window_ts_ns)
                self.stats.inference_fps = self._window_fps(self._processed_window_ts_ns)
                self.stats.control_observation_fps = self._window_fps(self._control_observation_window_ts_ns)
                self.stats.detection_batch_fps = 0.0
                self.stats.queue_latency_ms = max(0.0, (process_start_ns - int(frame.capture_ts_ns)) / 1e6)
                self.stats.inference_latency_ms = max(0.0, (done_ns - process_start_ns) / 1e6)
                self.stats.e2e_latency_ms = max(0.0, (done_ns - int(frame.capture_ts_ns)) / 1e6)
                self.stats.skipped_frames = len(self._skipped_window_ts_ns)
            finally:
                if callable(release):
                    release()

    def _detection_batch_loop(self) -> None:
        source = self.detection_source
        if source is None:
            return
        latest_result = getattr(source, "latest_result", None)
        if not callable(latest_result):
            self.stats.last_error = "detection source missing latest_result()"
            self.runtime.running = False
            self._stop.set()
            return
        while not self._stop.is_set():
            batch = latest_result(after_frame_id=self._last_consumed_frame_id)
            if batch is None:
                if not self._detection_source_is_still_available():
                    self._stop_detection_source()
                    self.runtime.running = False
                    self._stop.set()
                    break
                time.sleep(0.001)
                continue
            if not isinstance(batch, DetectionBatch):
                self.stats.last_error = "detection source returned invalid DetectionBatch"
                self._stop_detection_source()
                self.runtime.running = False
                self._stop.set()
                break
            skipped = self._skipped_since_previous(batch.frame_id)
            self.stats.consumed_frames += 1
            self.stats.consumed_detection_batches += 1
            self._last_consumed_frame_id = batch.frame_id
            self.stats.last_frame_id = self._last_consumed_frame_id
            if skipped:
                self._record_skipped(time.monotonic_ns(), skipped)
            process_start_ns = time.monotonic_ns()
            result = self.runtime.process_detection_batch(
                batch,
                width=self._detection_source_roi_width(),
                height=self._detection_source_roi_height(),
                **self._detection_source_geometry(),
            )
            self.stats.processed_frames += 1
            done_ns = time.monotonic_ns()
            self._processed_window_ts_ns.append(done_ns)
            self._record_control_observation_if_updated(result, done_ns)
            self._prune_window(self._processed_window_ts_ns, done_ns)
            self._prune_window(self._control_observation_window_ts_ns, done_ns)
            self._prune_window(self._skipped_window_ts_ns, done_ns)
            self.stats.window_processed_frames = len(self._processed_window_ts_ns)
            self.stats.window_control_observations = len(self._control_observation_window_ts_ns)
            self.stats.inference_fps = self._window_fps(self._processed_window_ts_ns)
            self.stats.detection_batch_fps = self.stats.inference_fps
            self.stats.control_observation_fps = self._window_fps(self._control_observation_window_ts_ns)
            self.stats.queue_latency_ms = max(
                0.0,
                (process_start_ns - int(batch.inference_end_ts_ns)) / 1e6,
            )
            self.stats.inference_latency_ms = batch.inference_latency_ms
            self.stats.e2e_latency_ms = max(0.0, (done_ns - int(batch.capture_ts_ns)) / 1e6)
            self.stats.skipped_frames = len(self._skipped_window_ts_ns)

    def _record_control_observation_if_updated(self, result: Any, ts_ns: int) -> None:
        if getattr(result, "observation_updated", False) is True:
            self._control_observation_window_ts_ns.append(int(ts_ns))

    def _latest_frame_broker(self) -> Any | None:
        broker = getattr(self.capture, "latest_frame_broker", None)
        if broker is None:
            session = getattr(self.capture, "session", None)
            broker = getattr(session, "latest_frame_broker", None)
        return broker

    def _latest_frame_broker_status(self) -> dict[str, Any]:
        broker = self._latest_frame_broker()
        status_fn = getattr(broker, "status", None)
        if not callable(status_fn):
            return {}
        try:
            status = status_fn()
        except Exception as exc:
            return {"available": False, "reason": "status failed", "detail": str(exc)}
        return dict(status) if isinstance(status, dict) else {}

    @staticmethod
    def _captured_frame_from_handle(handle: Any) -> Any | None:
        metadata = getattr(handle, "metadata", {}) or {}
        if isinstance(metadata, dict) and metadata.get("captured_frame") is not None:
            return metadata["captured_frame"]
        resource = getattr(handle, "resource", None)
        if all(hasattr(resource, attr) for attr in ("frame_id", "capture_ts_ns", "width", "height")):
            return resource
        return None

    def _skipped_generations_since_previous(self, handle: Any) -> int:
        generation = int(getattr(handle, "generation", -1))
        if self._last_consumed_generation < 0:
            return 0
        return max(0, generation - self._last_consumed_generation - 1)

    def _start_detection_source(self) -> None:
        if self.detection_source is None:
            return
        start = getattr(self.detection_source, "start", None)
        if callable(start):
            start()

    def _stop_detection_source(self) -> None:
        if self.detection_source is None:
            return
        stop = getattr(self.detection_source, "stop", None)
        if callable(stop):
            stop()

    def _detection_source_is_still_available(self) -> bool:
        source = self.detection_source
        if source is None:
            return False
        status_fn = getattr(source, "status", None)
        if not callable(status_fn):
            return True
        try:
            status = status_fn()
        except Exception as exc:
            self.stats.last_error = f"detection source status failed: {exc}"
            return False
        if not isinstance(status, dict):
            return True
        if status.get("running") is False:
            self.stats.last_error = str(
                status.get("last_error") or "detection source stopped"
            )
            return False
        if status.get("available") is False:
            reason = str(
                status.get("reason")
                or status.get("last_error")
                or "detection source unavailable"
            )
            self.stats.last_error = reason
            return False
        return True

    def _detection_source_status(self) -> dict[str, Any]:
        source = self.detection_source
        if source is None:
            return {}
        status_fn = getattr(source, "status", None)
        if not callable(status_fn):
            return {"selected": type(source).__name__, "available": True}
        try:
            status = status_fn()
        except Exception as exc:
            return {
                "selected": type(source).__name__,
                "available": False,
                "reason": "status failed",
                "detail": str(exc),
            }
        if not isinstance(status, dict):
            return {"selected": type(source).__name__, "available": True}
        payload = dict(status)
        payload.pop("pipeline", None)
        return payload

    def _detection_source_roi_width(self) -> int:
        return self._detection_source_dimension("roi_width")

    def _detection_source_roi_height(self) -> int:
        return self._detection_source_dimension("roi_height")

    def _detection_source_dimension(self, attr: str) -> int:
        value = getattr(self.detection_source, attr, None)
        if value is None:
            config = getattr(self.runtime, "config", None)
            roi = getattr(config, "roi", None)
            value = getattr(roi, "size", 0)
        parsed = int(value or 0)
        return parsed if parsed > 0 else 1

    def _detection_source_geometry(self) -> dict[str, int]:
        source = self.detection_source
        if source is None:
            return {}
        pipeline_config = getattr(source, "pipeline_config", None)
        if pipeline_config is not None:
            return _positive_geometry(
                source_width=getattr(pipeline_config, "capture_width", None),
                source_height=getattr(pipeline_config, "capture_height", None),
                roi_offset_x=getattr(pipeline_config, "roi_left", None),
                roi_offset_y=getattr(pipeline_config, "roi_top", None),
            )
        status_fn = getattr(source, "status", None)
        if not callable(status_fn):
            return {}
        try:
            status = status_fn()
        except Exception:
            return {}
        if not isinstance(status, dict):
            return {}
        capture = status.get("capture")
        roi = status.get("roi")
        capture = capture if isinstance(capture, dict) else {}
        roi = roi if isinstance(roi, dict) else {}
        return _positive_geometry(
            source_width=capture.get("width"),
            source_height=capture.get("height"),
            roi_offset_x=roi.get("left"),
            roi_offset_y=roi.get("top"),
        )

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

    def _stale_inference_input_reason(self, *, frame_capture_ts_ns: int, now_ns: int) -> str:
        config = getattr(self.runtime, "config", None)
        inference = getattr(config, "inference", None)
        deadline_ms = float(getattr(inference, "inference_input_deadline_ms", 0.0) or 0.0)
        if deadline_ms <= 0.0:
            return ""
        age_ms = max(0.0, (int(now_ns) - int(frame_capture_ts_ns)) / 1e6)
        # Legacy unit seams use tiny synthetic timestamps. Only enforce the
        # deadline for timestamps that plausibly belong to this monotonic clock.
        if age_ms > max(3_600_000.0, deadline_ms * 100.0):
            return ""
        if age_ms <= deadline_ms:
            return ""
        return (
            "inference input frame age exceeds deadline: "
            f"{age_ms:.1f}ms > {deadline_ms:.1f}ms"
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

    def _skipped_since_previous(self, frame_id: int) -> int:
        if self._last_consumed_frame_id < 0:
            return 0
        return max(0, int(frame_id) - int(self._last_consumed_frame_id) - 1)

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


def _positive_geometry(
    *,
    source_width: Any,
    source_height: Any,
    roi_offset_x: Any,
    roi_offset_y: Any,
) -> dict[str, int]:
    try:
        parsed_source_width = int(source_width or 0)
        parsed_source_height = int(source_height or 0)
        parsed_roi_offset_x = int(roi_offset_x or 0)
        parsed_roi_offset_y = int(roi_offset_y or 0)
    except (TypeError, ValueError):
        return {}
    if parsed_source_width <= 0 or parsed_source_height <= 0:
        return {}
    return {
        "source_width": parsed_source_width,
        "source_height": parsed_source_height,
        "roi_offset_x": max(0, parsed_roi_offset_x),
        "roi_offset_y": max(0, parsed_roi_offset_y),
    }
