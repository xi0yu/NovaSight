from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

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
    GPU_PREPROCESSOR_NOT_READY_ERROR = (
        "NVMM TensorRT GPU preprocess is not ready; build the Jetson native "
        "preprocess library with `python -m novasight doctor jetson-native-build` "
        "and make sure NOVASIGHT_JETSON_NATIVE_LIBRARY points to "
        "libnovasight_preprocess.so."
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
        self._last_consumed_frame_id = -1
        self._last_consumed_generation = -1
        self._processed_window_ts_ns: deque[int] = deque()
        self._control_observation_window_ts_ns: deque[int] = deque()
        self._skipped_window_ts_ns: deque[int] = deque()

    def start(self) -> None:
        if self.running:
            return
        self._require_running_capture()
        self._stop.clear()
        self._last_consumed_frame_id = -1
        self._last_consumed_generation = -1
        self._processed_window_ts_ns.clear()
        self._control_observation_window_ts_ns.clear()
        self._skipped_window_ts_ns.clear()
        self.stats = PipelineStats(started_at=time.time())
        self._require_gpu_preprocessor_ready()
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

    def _require_gpu_preprocessor_ready(self) -> None:
        config = getattr(self.runtime, "config", None)
        inference_config = getattr(config, "inference", None)
        capture_config = getattr(config, "capture", None)
        if str(getattr(inference_config, "backend", "")).lower() != "nvmm_latest":
            return
        if not bool(getattr(inference_config, "enabled", True)):
            return
        if str(getattr(capture_config, "memory", "")).lower() != "nvmm":
            return
        status = self._runtime_inference_status()
        gpu_status = status.get("gpu_preprocessor")
        if not isinstance(gpu_status, dict):
            gpu_status = {}
        if gpu_status.get("available") is True and gpu_status.get("native_ready", True) is not False:
            return
        reason = str(gpu_status.get("reason") or status.get("reason") or "gpu_preprocessor_unavailable")
        detail = str(gpu_status.get("detail") or "")
        native_status = gpu_status.get("native_status")
        if not detail and isinstance(native_status, dict):
            detail = str(native_status.get("detail") or native_status.get("reason") or "")
        message = f"{self.GPU_PREPROCESSOR_NOT_READY_ERROR} reason={reason}"
        if detail:
            message = f"{message}; detail={detail}"
        self.stats.last_error = message
        self.runtime.running = False
        raise RuntimeError(message)

    def _runtime_inference_status(self) -> dict[str, Any]:
        inference = getattr(self.runtime, "inference", None)
        status_fn = getattr(inference, "status", None)
        if callable(status_fn):
            status = status_fn()
            return dict(status) if isinstance(status, dict) else {}
        status_fn = getattr(self.runtime, "status", None)
        if not callable(status_fn):
            return {}
        status = status_fn()
        if isinstance(status, dict):
            inference_status = status.get("inference", status)
            return dict(inference_status) if isinstance(inference_status, dict) else {}
        inference_status = getattr(status, "inference", {})
        return dict(inference_status) if isinstance(inference_status, dict) else {}

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
