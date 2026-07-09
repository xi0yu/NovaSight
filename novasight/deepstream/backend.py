from __future__ import annotations

import importlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from novasight.contracts import DetectionBatch
from novasight.model_registry.manifest import ModelManifest

from .pipeline_builder import DeepStreamPipelineConfig, build_deepstream_pipeline
from .tensor_meta import output_tensor_to_detection_batch


GST_CLOCK_TIME_NONE = (1 << 64) - 1


@dataclass(frozen=True)
class DeepStreamDependencyStatus:
    available: bool
    reason: str = ""
    detail: str = ""


def check_deepstream_dependencies() -> DeepStreamDependencyStatus:
    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        Gst = importlib.import_module("gi.repository.Gst")
        init = getattr(Gst, "init", None)
        if callable(init):
            init(None)
    except Exception as exc:
        return DeepStreamDependencyStatus(
            available=False,
            reason="gstreamer-python-unavailable",
            detail=str(exc),
        )
    missing_elements = _missing_gst_elements(Gst)
    if missing_elements:
        return DeepStreamDependencyStatus(
            available=False,
            reason="deepstream-gst-elements-unavailable",
            detail="missing GStreamer elements: " + ", ".join(missing_elements),
        )
    try:
        importlib.import_module("pyds")
    except Exception as exc:
        return DeepStreamDependencyStatus(
            available=False,
            reason="pyds-unavailable",
            detail=str(exc),
        )
    return DeepStreamDependencyStatus(available=True)


def _missing_gst_elements(Gst: Any) -> list[str]:
    element_factory = getattr(Gst, "ElementFactory", None)
    find = getattr(element_factory, "find", None)
    if not callable(find):
        return []
    required = ("nvv4l2decoder", "nvvidconv", "nvstreammux", "nvinfer")
    return [name for name in required if find(name) is None]


class DeepStreamDetectionBackend:
    """Experimental DeepStream backend that publishes NovaSight DetectionBatch objects.

    The class is intentionally isolated from the legacy runtime. On non-Jetson
    development machines it reports missing GStreamer/pyds dependencies instead
    of silently falling back to CPU capture.
    """

    backend_id = "deepstream"

    def __init__(
        self,
        *,
        pipeline_config: DeepStreamPipelineConfig,
        manifest: ModelManifest,
        roi_width: int,
        roi_height: int,
        confidence_threshold: float | None = None,
        nms_threshold: float | None = None,
        max_publish_age_ms: float = 0.0,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.manifest = manifest
        self.roi_width = int(roi_width)
        self.roi_height = int(roi_height)
        self.confidence_threshold = (
            manifest.postprocess.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        self.nms_threshold = (
            manifest.postprocess.nms_iou_threshold
            if nms_threshold is None
            else float(nms_threshold)
        )
        self.max_publish_age_ms = max(0.0, float(max_publish_age_ms))
        self.pipeline_description = build_deepstream_pipeline(pipeline_config)
        self._lock = threading.RLock()
        self._pipeline: Any | None = None
        self._running = False
        self._last_result: DetectionBatch | None = None
        self._last_error = ""
        self._published_batches = 0
        self._stale_dropped_batches = 0
        self._tensor_meta_frames = 0
        self._postprocess_frames = 0
        self._publish_window_ts_ns: deque[int] = deque()
        self._stale_drop_window_ts_ns: deque[int] = deque()
        self._tensor_meta_window_ts_ns: deque[int] = deque()
        self._postprocess_window_ts_ns: deque[int] = deque()
        self._latency_window_samples: deque[tuple[int, float]] = deque()
        self._started_at_ns = 0
        self._gst_to_monotonic_offset_ns: int | None = None
        self._gst_base_time_ns: int | None = None
        self._pts_to_monotonic_offset_ns: int | None = None
        self._timestamp_source = "uninitialized"
        self._last_raw_pts_ns = 0
        self._last_capture_ts_ns = 0
        self._last_probe_observed_ts_ns = 0
        self._fallback_frame_id = 0
        self._bus_stop = threading.Event()
        self._bus_thread: threading.Thread | None = None
        self._terminal_error = False
        self._dependency_status: DeepStreamDependencyStatus | None = None

    def update_postprocess_thresholds(
        self,
        *,
        confidence_threshold: float,
        nms_threshold: float,
    ) -> None:
        with self._lock:
            self.confidence_threshold = float(confidence_threshold)
            self.nms_threshold = float(nms_threshold)

    def dependency_status(self, *, refresh: bool = False) -> DeepStreamDependencyStatus:
        with self._lock:
            cached = self._dependency_status
        if cached is not None and not refresh:
            return cached
        status = check_deepstream_dependencies()
        with self._lock:
            self._dependency_status = status
        return status

    def start(self) -> None:
        self.stop()
        dependency = self.dependency_status(refresh=True)
        if not dependency.available:
            self._last_error = dependency.detail or dependency.reason
            raise RuntimeError(f"DeepStream backend unavailable: {dependency.reason}: {dependency.detail}")
        Gst = importlib.import_module("gi.repository.Gst")
        Gst.init(None)
        pipeline = None
        try:
            pipeline = Gst.parse_launch(self.pipeline_description)
            self._attach_tensor_probe(pipeline)
            with self._lock:
                self._pipeline = pipeline
                self._running = True
                self._last_result = None
                self._published_batches = 0
                self._stale_dropped_batches = 0
                self._tensor_meta_frames = 0
                self._postprocess_frames = 0
                self._publish_window_ts_ns.clear()
                self._stale_drop_window_ts_ns.clear()
                self._tensor_meta_window_ts_ns.clear()
                self._postprocess_window_ts_ns.clear()
                self._latency_window_samples.clear()
                self._started_at_ns = time.monotonic_ns()
                self._gst_to_monotonic_offset_ns = None
                self._gst_base_time_ns = None
                self._pts_to_monotonic_offset_ns = None
                self._timestamp_source = "uninitialized"
                self._last_raw_pts_ns = 0
                self._last_capture_ts_ns = 0
                self._last_probe_observed_ts_ns = 0
                self._fallback_frame_id = 0
                self._last_error = ""
                self._terminal_error = False
            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("failed to set DeepStream pipeline to PLAYING")
        except Exception as exc:
            if pipeline is not None:
                _set_pipeline_null_best_effort(Gst, pipeline)
            with self._lock:
                if self._pipeline is pipeline:
                    self._pipeline = None
                self._running = False
                self._started_at_ns = 0
                self._publish_window_ts_ns.clear()
                self._stale_drop_window_ts_ns.clear()
                self._tensor_meta_window_ts_ns.clear()
                self._postprocess_window_ts_ns.clear()
                self._latency_window_samples.clear()
                self._last_error = str(exc)
            raise
        self._start_bus_monitor(pipeline)

    def stop(self) -> None:
        self._stop_bus_monitor()
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._running = False
            self._last_result = None
            self._publish_window_ts_ns.clear()
            self._tensor_meta_window_ts_ns.clear()
            self._postprocess_window_ts_ns.clear()
            self._latency_window_samples.clear()
            self._started_at_ns = 0
        if pipeline is None:
            return
        try:
            Gst = importlib.import_module("gi.repository.Gst")
            _set_pipeline_null_best_effort(Gst, pipeline)
        except Exception as exc:
            self._last_error = str(exc)

    def latest_result(self, after_frame_id: int | None = None) -> DetectionBatch | None:
        with self._lock:
            if self._terminal_error or not self._running:
                return None
            result = self._last_result
        if result is None:
            return None
        if after_frame_id is not None and result.frame_id <= int(after_frame_id):
            return None
        return result

    def status(self) -> dict[str, Any]:
        dependency = self.dependency_status()
        now_ns = time.monotonic_ns()
        with self._lock:
            running = self._running
            terminal_error = self._terminal_error
            if running and not terminal_error:
                self._prune_publish_window_locked(now_ns)
            last_result = self._last_result
            published = self._published_batches
            stale_dropped = self._stale_dropped_batches
            tensor_meta_frames = self._tensor_meta_frames
            postprocess_frames = self._postprocess_frames
            window_published = len(self._publish_window_ts_ns)
            window_stale_dropped = len(self._stale_drop_window_ts_ns)
            window_tensor_meta = len(self._tensor_meta_window_ts_ns)
            window_postprocess = len(self._postprocess_window_ts_ns)
            detection_batch_fps = self._publish_window_fps_locked()
            tensor_meta_fps = self._window_fps_locked(self._tensor_meta_window_ts_ns)
            postprocess_fps = self._window_fps_locked(self._postprocess_window_ts_ns)
            latency_stats = _latency_stats_locked(self._latency_window_samples)
            started_at_ns = self._started_at_ns
            last_error = self._last_error
            confidence_threshold = self.confidence_threshold
            nms_threshold = self.nms_threshold
            max_publish_age_ms = self.max_publish_age_ms
            timestamp_source = self._timestamp_source
            last_raw_pts_ns = self._last_raw_pts_ns
            last_capture_ts_ns = self._last_capture_ts_ns
            last_probe_observed_ts_ns = self._last_probe_observed_ts_ns
        uptime_ms = 0.0
        if started_at_ns:
            uptime_ms = max(0.0, (now_ns - started_at_ns) / 1e6)
        if not running or terminal_error:
            detection_batch_fps = 0.0
            tensor_meta_fps = 0.0
            postprocess_fps = 0.0
        last_frame_age_ms = 0.0
        last_inference_latency_ms = 0.0
        last_detection_count = 0
        if last_result is not None:
            last_frame_age_ms = max(0.0, (now_ns - int(last_result.capture_ts_ns)) / 1e6)
            last_inference_latency_ms = last_result.inference_latency_ms
            last_detection_count = len(last_result.detections)
        return {
            "selected": self.backend_id,
            "available": dependency.available,
            "reason": dependency.reason,
            "detail": dependency.detail,
            "running": running,
            "terminal_error": terminal_error,
            "model_id": self.manifest.model_id,
            "model_fingerprint": self.manifest.model_fingerprint,
            "model_input": {
                "name": self.manifest.input.name,
                "shape": list(self.manifest.input.shape),
                "dtype": self.manifest.input.dtype,
                "layout": self.manifest.input.layout,
                "color_format": self.manifest.input.color_format,
                "scale_factor": self.manifest.input.scale_factor,
                "maintain_aspect_ratio": self.manifest.input.maintain_aspect_ratio,
                "symmetric_padding": self.manifest.input.symmetric_padding,
            },
            "model_runtime": {
                "backend": self.manifest.runtime.backend,
                "precision": self.manifest.runtime.precision,
                "batch_size": self.manifest.runtime.batch_size,
            },
            "model_output": {
                "name": self.manifest.output.name,
                "shape": list(self.manifest.output.shape),
                "dtype": self.manifest.output.dtype,
                "class_count": self.manifest.output.class_count,
                "class_names": list(self.manifest.output.class_names),
            },
            "postprocess": {
                "parser": self.manifest.postprocess.parser,
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold,
                "class_aware_nms": self.manifest.postprocess.class_aware_nms,
            },
            "capture": {
                "device": self.pipeline_config.device,
                "pixel_format": self.pipeline_config.pixel_format,
                "width": self.pipeline_config.capture_width,
                "height": self.pipeline_config.capture_height,
                "fps": self.pipeline_config.fps,
                "io_mode": self.pipeline_config.io_mode,
                "batched_push_timeout_us": self.pipeline_config.batched_push_timeout_us,
            },
            "roi": {
                "left": self.pipeline_config.roi_left,
                "top": self.pipeline_config.roi_top,
                "size": self.pipeline_config.roi_size,
                "width": self.roi_width,
                "height": self.roi_height,
            },
            "published_batches": published,
            "window_published_batches": window_published if running and not terminal_error else 0,
            "stale_dropped_batches": stale_dropped,
            "window_stale_dropped_batches": window_stale_dropped if running and not terminal_error else 0,
            "max_publish_age_ms": max_publish_age_ms,
            "tensor_meta_frames": tensor_meta_frames,
            "postprocess_frames": postprocess_frames,
            "window_tensor_meta_frames": window_tensor_meta if running and not terminal_error else 0,
            "window_postprocess_frames": window_postprocess if running and not terminal_error else 0,
            "tensor_meta_fps": tensor_meta_fps,
            "postprocess_fps": postprocess_fps,
            "detection_batch_fps": detection_batch_fps,
            "last_frame_id": last_result.frame_id if last_result else 0,
            "last_frame_age_ms": last_frame_age_ms,
            "last_inference_latency_ms": last_inference_latency_ms,
            "capture_to_tensor_meta_ms_stats": latency_stats,
            "latency_source": "capture_to_tensor_meta_done",
            "timestamp_source": timestamp_source,
            "last_raw_pts_ns": last_raw_pts_ns,
            "last_capture_ts_ns": last_capture_ts_ns,
            "last_probe_observed_ts_ns": last_probe_observed_ts_ns,
            "last_pts_to_probe_ms": max(0.0, (last_probe_observed_ts_ns - last_capture_ts_ns) / 1e6)
            if last_probe_observed_ts_ns and last_capture_ts_ns
            else 0.0,
            "last_detection_count": last_detection_count,
            "last_error": last_error,
            "uptime_ms": uptime_ms,
            "pipeline": self.pipeline_description,
        }

    def _start_bus_monitor(self, pipeline: Any) -> None:
        self._bus_stop.clear()
        thread = threading.Thread(
            target=self._bus_monitor_loop,
            args=(pipeline,),
            name="novasight-deepstream-bus",
            daemon=True,
        )
        with self._lock:
            self._bus_thread = thread
        thread.start()

    def _stop_bus_monitor(self) -> None:
        self._bus_stop.set()
        with self._lock:
            thread = self._bus_thread
            self._bus_thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _bus_monitor_loop(self, pipeline: Any) -> None:
        try:
            Gst = importlib.import_module("gi.repository.Gst")
            bus = pipeline.get_bus()
            if bus is None:
                return
            mask = Gst.MessageType.ERROR | Gst.MessageType.EOS
            while not self._bus_stop.is_set():
                message = bus.timed_pop_filtered(100_000_000, mask)
                if message is None:
                    continue
                self._handle_bus_message(Gst, message)
                break
        except Exception as exc:
            self._record_terminal_error(f"DeepStream bus monitor failed: {exc}")

    def _handle_bus_message(self, Gst: Any, message: Any) -> None:
        message_type = getattr(message, "type", None)
        if message_type == getattr(Gst.MessageType, "ERROR", None):
            self._record_terminal_error(_format_gst_error(message))
            return
        if message_type == getattr(Gst.MessageType, "EOS", None):
            self._record_terminal_error("DeepStream pipeline reached EOS")

    def _record_terminal_error(self, reason: str, *, release_pipeline: bool = True) -> None:
        with self._lock:
            pipeline = self._pipeline if release_pipeline else None
            if release_pipeline:
                self._pipeline = None
            self._last_error = reason
            self._running = False
            self._terminal_error = True
        self._bus_stop.set()
        if pipeline is None:
            return
        try:
            Gst = importlib.import_module("gi.repository.Gst")
            _set_pipeline_null_best_effort(Gst, pipeline)
        except Exception:
            pass

    def publish_empty_detection_batch(
        self,
        *,
        frame_id: int,
        capture_ts_ns: int,
        inference_start_ts_ns: int,
        inference_end_ts_ns: int,
    ) -> DetectionBatch:
        with self._lock:
            timestamp_source = self._timestamp_source
        class_count = max(0, int(self.manifest.output.class_count))
        classes = list(getattr(self.manifest.output, "class_names", []) or [])
        if len(classes) < class_count:
            classes.extend(str(index) for index in range(len(classes), class_count))
        batch = DetectionBatch(
            frame_id=int(frame_id),
            capture_ts_ns=int(capture_ts_ns),
            inference_start_ts_ns=int(inference_start_ts_ns),
            inference_end_ts_ns=int(inference_end_ts_ns),
            detections=[],
            classes=classes[:class_count],
            coordinate_space="roi",
            metadata={
                "source": "deepstream",
                "timestamp_source": timestamp_source,
                "empty_reason": "missing_tensor_meta",
            },
        )
        self._publish_detection_batch(batch, tensor_meta=False, postprocess=False)
        return batch

    def publish_output_tensor(
        self,
        output: Any,
        *,
        frame_id: int,
        capture_ts_ns: int,
        inference_start_ts_ns: int,
        inference_end_ts_ns: int,
    ) -> DetectionBatch:
        with self._lock:
            confidence_threshold = self.confidence_threshold
            nms_threshold = self.nms_threshold
            timestamp_source = self._timestamp_source
        batch = output_tensor_to_detection_batch(
            output,
            manifest=self.manifest,
            frame_id=frame_id,
            capture_ts_ns=capture_ts_ns,
            inference_start_ts_ns=inference_start_ts_ns,
            inference_end_ts_ns=inference_end_ts_ns,
            roi_width=self.roi_width,
            roi_height=self.roi_height,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            metadata={
                "source": "deepstream",
                "timestamp_source": timestamp_source,
            },
        )
        self._publish_detection_batch(batch, tensor_meta=True, postprocess=True)
        return batch

    def _publish_detection_batch(
        self,
        batch: DetectionBatch,
        *,
        tensor_meta: bool,
        postprocess: bool,
    ) -> None:
        with self._lock:
            if self._terminal_error or not self._running:
                return
            inference_end_ts_ns = int(batch.inference_end_ts_ns)
            self._latency_window_samples.append(
                (inference_end_ts_ns, float(batch.inference_latency_ms))
            )
            if tensor_meta:
                self._tensor_meta_frames += 1
                self._tensor_meta_window_ts_ns.append(inference_end_ts_ns)
            if postprocess:
                self._postprocess_frames += 1
                self._postprocess_window_ts_ns.append(inference_end_ts_ns)
            if self._batch_age_exceeds_publish_limit_locked(batch):
                self._record_stale_drop_locked(
                    inference_end_ts_ns,
                    (
                        "DeepStream DetectionBatch dropped before runtime: "
                        f"age {self._batch_age_ms(batch):.1f}ms > {self.max_publish_age_ms:.1f}ms"
                    ),
                )
                return
            self._last_result = batch
            self._published_batches += 1
            self._publish_window_ts_ns.append(inference_end_ts_ns)
            self._prune_publish_window_locked(inference_end_ts_ns)
            if not self._terminal_error:
                self._last_error = ""

    def _batch_age_exceeds_publish_limit_locked(self, batch: DetectionBatch) -> bool:
        if self.max_publish_age_ms <= 0.0:
            return False
        return self._batch_age_ms(batch) > self.max_publish_age_ms

    @staticmethod
    def _batch_age_ms(batch: DetectionBatch) -> float:
        try:
            return max(
                0.0,
                (int(batch.inference_end_ts_ns) - int(batch.capture_ts_ns)) / 1e6,
            )
        except Exception:
            return 0.0

    def _record_stale_drop_locked(self, timestamp_ns: int, reason: str) -> None:
        self._stale_dropped_batches += 1
        self._stale_drop_window_ts_ns.append(int(timestamp_ns))
        self._prune_publish_window_locked(int(timestamp_ns))
        self._last_error = reason

    def _publish_window_fps_locked(self) -> float:
        return self._window_fps_locked(self._publish_window_ts_ns)

    def _window_fps_locked(self, timestamps_ns: deque[int]) -> float:
        if len(timestamps_ns) < 2:
            return 0.0
        elapsed_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        if elapsed_s <= 0:
            return 0.0
        return (len(timestamps_ns) - 1) / elapsed_s

    def _prune_publish_window_locked(self, now_ns: int) -> None:
        window_start_ns = int(now_ns) - 1_000_000_000
        while self._publish_window_ts_ns and self._publish_window_ts_ns[0] < window_start_ns:
            self._publish_window_ts_ns.popleft()
        while self._stale_drop_window_ts_ns and self._stale_drop_window_ts_ns[0] < window_start_ns:
            self._stale_drop_window_ts_ns.popleft()
        while self._tensor_meta_window_ts_ns and self._tensor_meta_window_ts_ns[0] < window_start_ns:
            self._tensor_meta_window_ts_ns.popleft()
        while self._postprocess_window_ts_ns and self._postprocess_window_ts_ns[0] < window_start_ns:
            self._postprocess_window_ts_ns.popleft()
        while self._latency_window_samples and self._latency_window_samples[0][0] < window_start_ns:
            self._latency_window_samples.popleft()

    def _attach_tensor_probe(self, pipeline: Any) -> None:
        nvinfer = pipeline.get_by_name("primary-infer")
        if nvinfer is None:
            raise RuntimeError("DeepStream pipeline missing primary-infer element")
        sink_pad = nvinfer.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError("DeepStream primary-infer element has no sink pad")
        src_pad = nvinfer.get_static_pad("src")
        if src_pad is None:
            raise RuntimeError("DeepStream primary-infer element has no src pad")
        Gst = importlib.import_module("gi.repository.Gst")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._drop_stale_input_probe)
        src_pad.add_probe(Gst.PadProbeType.BUFFER, self._tensor_probe)

    def _drop_stale_input_probe(self, _pad: Any, info: Any) -> Any:
        Gst = importlib.import_module("gi.repository.Gst")
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            observed_ns = time.monotonic_ns()
            raw_pts_ns = _buffer_pts_ns(buffer)
            capture_ts_ns = self._capture_ts_from_pts(raw_pts_ns, observed_ns=observed_ns)
            age_ms = max(0.0, (int(observed_ns) - int(capture_ts_ns)) / 1e6)
            with self._lock:
                self._last_raw_pts_ns = raw_pts_ns
                self._last_capture_ts_ns = int(capture_ts_ns)
                self._last_probe_observed_ts_ns = int(observed_ns)
                if (
                    self._terminal_error
                    or not self._running
                    or self.max_publish_age_ms <= 0.0
                    or age_ms <= self.max_publish_age_ms
                ):
                    return Gst.PadProbeReturn.OK
                self._record_stale_drop_locked(
                    observed_ns,
                    f"DeepStream buffer dropped before nvinfer: age {age_ms:.1f}ms > {self.max_publish_age_ms:.1f}ms",
                )
                return Gst.PadProbeReturn.DROP
        except Exception as exc:
            self._record_terminal_error(
                f"DeepStream stale input probe failed: {exc}",
                release_pipeline=False,
            )
            return Gst.PadProbeReturn.OK

    def _tensor_probe(self, _pad: Any, info: Any) -> Any:
        Gst = importlib.import_module("gi.repository.Gst")
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            for frame_id, capture_ts_ns, output in self._iter_output_tensors(buffer):
                done_ns = time.monotonic_ns()
                raw_pts_ns = int(capture_ts_ns or 0)
                capture_ts_ns = self._capture_ts_from_pts(raw_pts_ns, observed_ns=done_ns)
                with self._lock:
                    self._last_raw_pts_ns = raw_pts_ns
                    self._last_capture_ts_ns = int(capture_ts_ns)
                    self._last_probe_observed_ts_ns = int(done_ns)
                if output is None:
                    self.publish_empty_detection_batch(
                        frame_id=frame_id,
                        capture_ts_ns=capture_ts_ns,
                        inference_start_ts_ns=capture_ts_ns,
                        inference_end_ts_ns=done_ns,
                    )
                else:
                    self.publish_output_tensor(
                        output,
                        frame_id=frame_id,
                        capture_ts_ns=capture_ts_ns,
                        inference_start_ts_ns=capture_ts_ns,
                        inference_end_ts_ns=done_ns,
                    )
        except Exception as exc:
            self._record_terminal_error(
                f"DeepStream tensor probe failed: {exc}",
                release_pipeline=False,
            )
        return Gst.PadProbeReturn.OK

    def _capture_ts_from_pts(self, pts_ns: int, *, observed_ns: int) -> int:
        pts_ns = int(pts_ns or 0)
        observed_ns = int(observed_ns)
        if pts_ns <= 0 or pts_ns >= GST_CLOCK_TIME_NONE:
            with self._lock:
                self._timestamp_source = "observed_probe_time_invalid_pts"
            return observed_ns
        capture_ts_ns = self._capture_ts_from_pipeline_clock(pts_ns, observed_ns=observed_ns)
        if capture_ts_ns is not None:
            with self._lock:
                self._timestamp_source = "gst_clock_base_time_pts"
            return capture_ts_ns
        with self._lock:
            if self._pts_to_monotonic_offset_ns is None:
                self._pts_to_monotonic_offset_ns = observed_ns - pts_ns
            self._timestamp_source = "first_probe_offset_pts"
            return pts_ns + self._pts_to_monotonic_offset_ns

    def _capture_ts_from_pipeline_clock(self, pts_ns: int, *, observed_ns: int) -> int | None:
        with self._lock:
            pipeline = self._pipeline
            offset_ns = self._gst_to_monotonic_offset_ns
            base_time_ns = self._gst_base_time_ns
        if pipeline is None:
            return None
        try:
            clock = pipeline.get_clock()
            if clock is None:
                return None
            if offset_ns is None or base_time_ns is None:
                gst_now_ns = int(clock.get_time())
                base_time_ns = int(pipeline.get_base_time())
                if gst_now_ns <= 0 or base_time_ns <= 0:
                    return None
                offset_ns = int(observed_ns) - gst_now_ns
                with self._lock:
                    self._gst_to_monotonic_offset_ns = offset_ns
                    self._gst_base_time_ns = base_time_ns
            return int(base_time_ns) + int(pts_ns) + int(offset_ns)
        except Exception:
            return None

    def _frame_id_from_meta(self, frame_meta: Any) -> int:
        frame_num = getattr(frame_meta, "frame_num", None)
        if frame_num is not None:
            return int(frame_num)
        with self._lock:
            frame_id = self._fallback_frame_id
            self._fallback_frame_id += 1
            return frame_id

    def _iter_output_tensors(self, buffer: Any) -> Any:
        """Yield output tensors from NvDsInferTensorMeta.

        This method is kept small and defensive because it can only be fully
        exercised on Jetson with DeepStream's pyds module installed.
        """
        pyds = importlib.import_module("pyds")
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return
        tensor_meta_type = getattr(
            getattr(pyds, "NvDsMetaType", object),
            "NVDSINFER_TENSOR_OUTPUT_META",
            None,
        )
        if tensor_meta_type is None:
            raise RuntimeError("DeepStream pyds missing NVDSINFER_TENSOR_OUTPUT_META")
        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            frame_id = self._frame_id_from_meta(frame_meta)
            pts = int(getattr(frame_meta, "buf_pts", 0) or getattr(buffer, "pts", 0) or time.monotonic_ns())
            user_meta_list = frame_meta.frame_user_meta_list
            yielded_tensor = False
            while user_meta_list is not None:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                meta_type = getattr(user_meta, "base_meta", None)
                meta_type_value = getattr(meta_type, "meta_type", None)
                if meta_type_value == tensor_meta_type:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    output = _tensor_meta_output_to_numpy(
                        pyds,
                        tensor_meta,
                        output_name=self.manifest.output.name,
                        dtype=self.manifest.output.dtype,
                    )
                    if output is not None:
                        yielded_tensor = True
                        yield frame_id, pts, output
                user_meta_list = user_meta_list.next
            if not yielded_tensor:
                yield frame_id, pts, None
            frame_list = frame_list.next


def _tensor_meta_first_output_to_numpy(pyds: Any, tensor_meta: Any, *, dtype: str = "float32") -> Any:
    return _tensor_meta_output_to_numpy(
        pyds,
        tensor_meta,
        output_name="",
        dtype=dtype,
    )


def _tensor_meta_output_to_numpy(
    pyds: Any,
    tensor_meta: Any,
    *,
    output_name: str,
    dtype: str = "float32",
) -> Any:
    output_name = str(output_name or "").strip()
    layer_count = int(getattr(tensor_meta, "num_output_layers", 1) or 1)
    fallback = None
    for index in range(max(1, layer_count)):
        layer = pyds.get_nvds_LayerInfo(tensor_meta, index)
        if output_name and not _layer_name_matches(layer, output_name):
            if fallback is None and layer_count == 1 and not _layer_name(layer):
                fallback = _tensor_layer_to_numpy(pyds, layer, dtype=dtype)
            continue
        return _tensor_layer_to_numpy(pyds, layer, dtype=dtype)
    if fallback is not None:
        return fallback
    if output_name:
        raise ValueError(f"DeepStream tensor output layer not found: {output_name}")
    return None


def _tensor_layer_to_numpy(pyds: Any, layer: Any, *, dtype: str = "float32") -> Any:
    import ctypes

    import numpy as np

    dims = getattr(layer, "inferDims", None)
    shape = _layer_shape(dims)
    if not shape:
        return None
    np_dtype, c_type = _tensor_buffer_dtype(dtype, ctypes=ctypes, np=np)
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(c_type))
    array = np.ctypeslib.as_array(ptr, shape=tuple(shape))
    if np_dtype == np.float16:
        array = array.view(np.float16)
    return np.asarray(array, dtype=np.float32).copy()


def _layer_name_matches(layer: Any, expected: str) -> bool:
    actual = _layer_name(layer)
    return bool(actual) and actual == expected


def _layer_name(layer: Any) -> str:
    for attr in ("layerName", "layer_name", "name"):
        value = getattr(layer, attr, None)
        if value is None:
            continue
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    return ""


def _layer_shape(dims: Any) -> list[int]:
    if dims is None:
        return []
    if hasattr(dims, "d") and hasattr(dims, "numDims"):
        values = [int(dims.d[index]) for index in range(int(dims.numDims))]
        return values
    if hasattr(dims, "numElements"):
        return [int(dims.numElements)]
    return []


def _tensor_buffer_dtype(dtype: str, *, ctypes: Any, np: Any) -> tuple[Any, Any]:
    normalized = str(dtype or "float32").strip().lower().replace("-", "").replace("_", "")
    if normalized in {"fp32", "float", "float32", "f32"}:
        return np.float32, ctypes.c_float
    if normalized in {"fp16", "float16", "half", "f16"}:
        return np.float16, ctypes.c_uint16
    raise ValueError(f"unsupported DeepStream tensor output dtype: {dtype}")


def _latency_stats_locked(samples: deque[tuple[int, float]]) -> dict[str, float | int]:
    values = sorted(float(value) for _timestamp_ns, value in samples)
    if not values:
        return {
            "count": 0,
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": _nearest_rank_percentile(values, 50.0),
        "p95": _nearest_rank_percentile(values, 95.0),
        "p99": _nearest_rank_percentile(values, 99.0),
        "max": values[-1],
    }


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = int(round((float(percentile) / 100.0) * (len(values) - 1)))
    rank = max(0, min(len(values) - 1, rank))
    return values[rank]


def _buffer_pts_ns(buffer: Any) -> int:
    value = getattr(buffer, "pts", None)
    if value is None:
        get_pts = getattr(buffer, "get_pts", None)
        if callable(get_pts):
            value = get_pts()
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _format_gst_error(message: Any) -> str:
    parse_error = getattr(message, "parse_error", None)
    if not callable(parse_error):
        return "DeepStream pipeline error"
    try:
        error, debug = parse_error()
    except Exception as exc:
        return f"DeepStream pipeline error: {exc}"
    parts = ["DeepStream pipeline error"]
    if error:
        parts.append(str(error))
    if debug:
        parts.append(str(debug))
    return ": ".join(parts)


def _set_pipeline_null_best_effort(Gst: Any, pipeline: Any) -> None:
    try:
        pipeline.set_state(Gst.State.NULL)
    except Exception:
        pass
