from __future__ import annotations

import ctypes
import importlib
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novasight.contracts import BBox, Detection, DetectionBatch
from novasight.model_registry.manifest import ModelManifest
from novasight.detection_batch_mailbox import DetectionBatchMailbox

from .pipeline_builder import DeepStreamPipelineConfig, build_deepstream_pipeline
from .parser_build import ensure_deepstream_parser_library


GST_CLOCK_TIME_NONE = (1 << 64) - 1
logger = logging.getLogger("novasight.deepstream.backend")


@dataclass(frozen=True, slots=True)
class DeepStreamDependencyStatus:
    available: bool
    reason: str = ""
    detail: str = ""


class _ParserTelemetry:
    def __init__(self, library_path: Path) -> None:
        self._library: Any | None = None
        self._error = ""
        try:
            library = ctypes.CDLL(str(library_path))
            for name in (
                "novasight_parser_decode_calls",
                "novasight_parser_last_decode_ns",
                "novasight_parser_parse_failures",
                "novasight_parser_last_error_code",
                "novasight_parser_last_input_candidates",
                "novasight_parser_last_output_candidates",
            ):
                function = getattr(library, name)
                function.argtypes = []
                function.restype = ctypes.c_uint64
            reset = getattr(library, "novasight_parser_reset_telemetry")
            reset.argtypes = []
            reset.restype = None
            self._library = library
        except Exception as exc:
            self._error = str(exc)

    def reset(self) -> None:
        if self._library is not None:
            self._library.novasight_parser_reset_telemetry()

    def snapshot(self) -> dict[str, object]:
        library = self._library
        if library is None:
            return {"available": False, "reason": self._error or "parser telemetry unavailable"}
        return {
            "available": True,
            "decode_calls": int(library.novasight_parser_decode_calls()),
            "decode_ms": float(library.novasight_parser_last_decode_ns()) / 1e6,
            "parse_failures": int(library.novasight_parser_parse_failures()),
            "last_error_code": int(library.novasight_parser_last_error_code()),
            "input_candidates": int(library.novasight_parser_last_input_candidates()),
            "decoded_candidates": int(library.novasight_parser_last_output_candidates()),
        }


class DeepStreamObjectBackend:
    """NVMM/nvinfer detection source that publishes object-meta batches only."""

    backend_id = "deepstream_nvinfer"

    def __init__(
        self,
        *,
        pipeline_config: DeepStreamPipelineConfig,
        manifest: ModelManifest,
        parser_library_path: Path,
        max_publish_age_ms: float,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.manifest = manifest
        self.parser_library_path = Path(parser_library_path).expanduser().resolve(strict=False)
        self.max_publish_age_ms = max(0.0, float(max_publish_age_ms))
        self.pipeline_description = build_deepstream_pipeline(pipeline_config)
        self.detection_batch_mailbox = DetectionBatchMailbox()
        self._lock = threading.RLock()
        self._pipeline: Any | None = None
        self._running = False
        self._terminal_error = False
        self._last_error = ""
        self._bus_stop = threading.Event()
        self._bus_thread: threading.Thread | None = None
        self._dependency_status: DeepStreamDependencyStatus | None = None
        self._parser_telemetry = _ParserTelemetry(self.parser_library_path)
        self._parser_auto_build: dict[str, object] = {
            "attempted": False,
            "success": self.parser_library_path.is_file(),
            "detail": (
                "existing library" if self.parser_library_path.is_file() else "not attempted"
            ),
        }
        self._inference_start_by_pts: OrderedDict[int, int] = OrderedDict()
        self._gst_to_monotonic_offset_ns: int | None = None
        self._gst_base_time_ns: int | None = None
        self._fallback_pts_offset_ns: int | None = None
        self._timestamp_source = "uninitialized"
        self._generation = -1
        self._last_frame_id = -1
        self._last_capture_ts_ns = 0
        self._last_capture_interval_ms = 0.0
        self._input_frames = 0
        self._published_batches = 0
        self._stale_dropped_batches = 0
        self._non_monotonic_dropped_batches = 0
        self._object_meta_frames = 0
        self._last_batch_age_ms = 0.0
        self._input_frame_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._publish_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._input_age_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._inference_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._build_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._started_at_ns = 0

    def dependency_status(self, *, refresh: bool = False) -> DeepStreamDependencyStatus:
        with self._lock:
            cached = self._dependency_status
        if cached is not None and not refresh:
            return cached
        status = _check_dependencies(self.parser_library_path)
        with self._lock:
            self._dependency_status = status
        return status

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running and not self._terminal_error

    def start(self) -> None:
        self.stop()
        self._ensure_parser_library()
        self._parser_telemetry.reset()
        dependency = self.dependency_status(refresh=True)
        if not dependency.available:
            self._last_error = dependency.detail or dependency.reason
            raise RuntimeError(
                f"DeepStream backend unavailable: {dependency.reason}: {dependency.detail}"
            )
        Gst = importlib.import_module("gi.repository.Gst")
        Gst.init(None)
        pipeline = None
        try:
            pipeline = Gst.parse_launch(self.pipeline_description)
            with self._lock:
                self._reset_state_locked()
                self._pipeline = pipeline
                self._running = True
                self._started_at_ns = time.monotonic_ns()
            self._attach_probes(Gst, pipeline)
            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("failed to set DeepStream pipeline to PLAYING")
        except Exception as exc:
            if pipeline is not None:
                _set_pipeline_null_best_effort(Gst, pipeline)
            with self._lock:
                self._pipeline = None
                self._running = False
                self._last_error = str(exc)
            raise
        self._start_bus_monitor(pipeline)

    def stop(self) -> None:
        self._stop_bus_monitor()
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._running = False
            self._inference_start_by_pts.clear()
            self.detection_batch_mailbox.clear()
        if pipeline is None:
            return
        try:
            Gst = importlib.import_module("gi.repository.Gst")
            _set_pipeline_null_best_effort(Gst, pipeline)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)

    def status(self) -> dict[str, object]:
        dependency = self.dependency_status()
        now_ns = time.monotonic_ns()
        with self._lock:
            self._prune_samples_locked(now_ns)
            running = self._running and not self._terminal_error
            uptime_ms = (
                max(0.0, (now_ns - self._started_at_ns) / 1e6)
                if self._started_at_ns
                else 0.0
            )
            payload: dict[str, object] = {
                "selected": self.backend_id,
                "available": dependency.available,
                "reason": dependency.reason,
                "detail": dependency.detail,
                "running": running,
                "terminal_error": self._terminal_error,
                "last_error": self._last_error,
                "pipeline": self.pipeline_description,
                "timestamp_source": self._timestamp_source,
                "clock_domain": "monotonic",
                "input_frames": self._input_frames,
                "input_fps": _sample_rate(self._input_frame_samples, now_ns),
                "published_batches": self._published_batches,
                "published_fps": _sample_rate(self._publish_samples, now_ns),
                "stale_dropped_batches": self._stale_dropped_batches,
                "max_publish_age_ms": self.max_publish_age_ms,
                "non_monotonic_dropped_batches": self._non_monotonic_dropped_batches,
                "object_meta_frames": self._object_meta_frames,
                "python_nms": False,
                "postprocess_owner": "native_parser_then_deepstream_cluster_mode_2",
                "parser_library": str(self.parser_library_path),
                "parser_auto_build": dict(self._parser_auto_build),
                "last_frame_id": self._last_frame_id,
                "last_capture_ts_ns": self._last_capture_ts_ns,
                "latest_frame_age_ms": (
                    max(0.0, (now_ns - self._last_capture_ts_ns) / 1e6)
                    if self._last_capture_ts_ns > 0
                    else 0.0
                ),
                "last_capture_interval_ms": self._last_capture_interval_ms,
                "last_batch_age_ms": self._last_batch_age_ms,
                "capture_profile": {
                    "device": self.pipeline_config.device,
                    "pixel_format": self.pipeline_config.pixel_format,
                    "width": self.pipeline_config.capture_width,
                    "height": self.pipeline_config.capture_height,
                    "fps": self.pipeline_config.fps,
                },
                "roi_region": {
                    "x": self.pipeline_config.roi_left,
                    "y": self.pipeline_config.roi_top,
                    "width": self.pipeline_config.roi_width,
                    "height": self.pipeline_config.roi_height,
                },
                "nvmm_output": {
                    "width": self.pipeline_config.model_width,
                    "height": self.pipeline_config.model_height,
                    "pixel_format": "NV12",
                    "memory": "NVMM",
                },
                "model_input": {
                    "width": self.pipeline_config.model_width,
                    "height": self.pipeline_config.model_height,
                },
                "batch_age_ms_stats": _sample_stats(self._publish_samples),
                "inference_input_age_ms_stats": _sample_stats(self._input_age_samples),
                "nvinfer_total_ms_stats": _sample_stats(self._inference_samples),
                "detection_batch_build_ms_stats": _sample_stats(self._build_samples),
                "output_sync_copy_ms": None,
                "output_sync_copy_reason": "nvinfer does not expose per-frame copy timing",
                "nms_ms": None,
                "nms_timing_reason": "DeepStream cluster-mode=2 has no per-frame NMS timing API",
                "parser": self._parser_telemetry.snapshot(),
                "detection_batch_mailbox": self.detection_batch_mailbox.status(),
                "uptime_ms": uptime_ms,
            }
        return payload

    def _ensure_parser_library(self) -> None:
        if self.parser_library_path.is_file():
            return
        self._parser_auto_build = {
            "attempted": True,
            "success": False,
            "detail": "building",
        }
        try:
            ensure_deepstream_parser_library(self.parser_library_path)
        except Exception as exc:
            detail = str(exc)
            logger.exception("DeepStream parser automatic build failed")
            self._parser_auto_build = {
                "attempted": True,
                "success": False,
                "detail": detail,
            }
            with self._lock:
                self._dependency_status = DeepStreamDependencyStatus(
                    False,
                    "deepstream-parser-auto-build-failed",
                    detail,
                )
                self._last_error = detail
            raise RuntimeError(f"DeepStream parser automatic build failed: {detail}") from exc
        self._parser_telemetry = _ParserTelemetry(self.parser_library_path)
        self._parser_auto_build = {
            "attempted": True,
            "success": True,
            "detail": str(self.parser_library_path),
        }
        with self._lock:
            self._dependency_status = None

    def _reset_state_locked(self) -> None:
        self.detection_batch_mailbox.clear()
        self._terminal_error = False
        self._last_error = ""
        self._inference_start_by_pts.clear()
        self._gst_to_monotonic_offset_ns = None
        self._gst_base_time_ns = None
        self._fallback_pts_offset_ns = None
        self._timestamp_source = "uninitialized"
        self._generation = -1
        self._last_frame_id = -1
        self._last_capture_ts_ns = 0
        self._last_capture_interval_ms = 0.0
        self._input_frames = 0
        self._published_batches = 0
        self._stale_dropped_batches = 0
        self._non_monotonic_dropped_batches = 0
        self._object_meta_frames = 0
        self._last_batch_age_ms = 0.0
        self._input_frame_samples.clear()
        self._publish_samples.clear()
        self._input_age_samples.clear()
        self._inference_samples.clear()
        self._build_samples.clear()

    def _attach_probes(self, Gst: Any, pipeline: Any) -> None:
        nvinfer = pipeline.get_by_name("primary-infer")
        if nvinfer is None:
            raise RuntimeError("DeepStream pipeline missing primary-infer")
        sink_pad = nvinfer.get_static_pad("sink")
        src_pad = nvinfer.get_static_pad("src")
        if sink_pad is None or src_pad is None:
            raise RuntimeError("DeepStream primary-infer pads are unavailable")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._inference_start_probe)
        src_pad.add_probe(Gst.PadProbeType.BUFFER, self._object_meta_probe)

    def _inference_start_probe(self, _pad: Any, info: Any) -> Any:
        Gst = importlib.import_module("gi.repository.Gst")
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            pts_ns = _buffer_pts_ns(buffer)
            now_ns = time.monotonic_ns()
            capture_ts_ns = self._capture_ts_from_pts(pts_ns, observed_ns=now_ns)
            age_ms = max(0.0, (now_ns - capture_ts_ns) / 1e6)
            with self._lock:
                if self._terminal_error or not self._running:
                    return Gst.PadProbeReturn.DROP
                if self.max_publish_age_ms > 0.0 and age_ms > self.max_publish_age_ms:
                    self._stale_dropped_batches += 1
                    self._last_error = (
                        "DeepStream buffer dropped before nvinfer: "
                        f"age {age_ms:.1f}ms > {self.max_publish_age_ms:.1f}ms"
                    )
                    return Gst.PadProbeReturn.DROP
                self._inference_start_by_pts[pts_ns] = now_ns
                self._inference_start_by_pts.move_to_end(pts_ns)
                self._input_frames += 1
                self._input_frame_samples.append((now_ns, 0.0))
                while len(self._inference_start_by_pts) > 32:
                    self._inference_start_by_pts.popitem(last=False)
            return Gst.PadProbeReturn.OK
        except Exception as exc:
            self._record_terminal_error(f"DeepStream nvinfer sink probe failed: {exc}")
            return Gst.PadProbeReturn.DROP

    def _object_meta_probe(self, _pad: Any, info: Any) -> Any:
        Gst = importlib.import_module("gi.repository.Gst")
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            pyds = importlib.import_module("pyds")
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch_meta is None:
                raise RuntimeError("nvinfer output buffer has no NvDsBatchMeta")
            frame_list = batch_meta.frame_meta_list
            while frame_list is not None:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                self._publish_frame_meta(pyds, frame_meta, buffer)
                frame_list = frame_list.next
        except Exception as exc:
            self._record_terminal_error(f"DeepStream object-meta probe failed: {exc}")
        return Gst.PadProbeReturn.OK

    def _publish_frame_meta(self, pyds: Any, frame_meta: Any, buffer: Any) -> None:
        inference_end_ts_ns = time.monotonic_ns()
        raw_pts_value = getattr(frame_meta, "buf_pts", None)
        raw_pts_ns = int(
            _buffer_pts_ns(buffer) if raw_pts_value is None else raw_pts_value
        )
        capture_ts_ns = self._capture_ts_from_pts(raw_pts_ns, observed_ns=inference_end_ts_ns)
        with self._lock:
            inference_start_ts_ns = self._inference_start_by_pts.pop(
                raw_pts_ns,
                capture_ts_ns,
            )
        build_start_ns = time.monotonic_ns()
        detections = self._object_meta_detections(pyds, frame_meta)
        publish_ts_ns = time.monotonic_ns()
        frame_id_value = getattr(frame_meta, "frame_num", None)
        frame_id = int(self._last_frame_id + 1 if frame_id_value is None else frame_id_value)
        parser = self._parser_telemetry.snapshot()
        input_age_ms = max(0.0, (inference_start_ts_ns - capture_ts_ns) / 1e6)
        inference_ms = max(0.0, (inference_end_ts_ns - inference_start_ts_ns) / 1e6)
        batch_age_ms = max(0.0, (publish_ts_ns - capture_ts_ns) / 1e6)
        build_ms = max(0.0, (publish_ts_ns - build_start_ns) / 1e6)
        with self._lock:
            if self._terminal_error or not self._running:
                return
            if self._timestamp_source != "gst_clock_base_time_pts":
                self._non_monotonic_dropped_batches += 1
                self._last_error = (
                    "DeepStream timestamp is not mapped through pipeline clock/base-time: "
                    f"{self._timestamp_source}"
                )
                return
            if frame_id <= self._last_frame_id or capture_ts_ns <= self._last_capture_ts_ns:
                self._non_monotonic_dropped_batches += 1
                self._last_error = "DeepStream frame identity or capture timestamp is non-monotonic"
                return
            if self.max_publish_age_ms > 0.0 and batch_age_ms > self.max_publish_age_ms:
                self._stale_dropped_batches += 1
                self._last_error = (
                    "DeepStream DetectionBatch dropped before control: "
                    f"age {batch_age_ms:.1f}ms > {self.max_publish_age_ms:.1f}ms"
                )
                return
            self._generation += 1
            generation = self._generation
        classes = list(self.manifest.output.class_names)
        batch = DetectionBatch(
            frame_id=frame_id,
            generation=generation,
            source_sequence=frame_id,
            capture_ts_ns=capture_ts_ns,
            inference_start_ts_ns=inference_start_ts_ns,
            inference_end_ts_ns=inference_end_ts_ns,
            publish_ts_ns=publish_ts_ns,
            input_age_ms=input_age_ms,
            inference_ms=inference_ms,
            result_age_ms=batch_age_ms,
            detections=detections,
            classes=classes,
            coordinate_space="roi",
            clock_domain="monotonic",
            roi_size=(self.pipeline_config.roi_width, self.pipeline_config.roi_height),
            model_input_size=(
                self.pipeline_config.model_width,
                self.pipeline_config.model_height,
            ),
            metadata={
                "source": self.backend_id,
                "timestamp_source": self._timestamp_source,
                "raw_pts_ns": raw_pts_ns,
                "postprocess_owner": "nvinfer_custom_parser_and_cluster_mode_2",
                "python_nms": False,
                "parser": parser,
                "nvinfer_total_ms": inference_ms,
                "detection_batch_build_ms": build_ms,
                "roi_offset_x": self.pipeline_config.roi_left,
                "roi_offset_y": self.pipeline_config.roi_top,
                "source_width": self.pipeline_config.capture_width,
                "source_height": self.pipeline_config.capture_height,
            },
        )
        with self._lock:
            if not self.detection_batch_mailbox.publish(batch):
                self._non_monotonic_dropped_batches += 1
                return
            if self._last_capture_ts_ns > 0:
                self._last_capture_interval_ms = max(
                    0.0,
                    (capture_ts_ns - self._last_capture_ts_ns) / 1e6,
                )
            self._last_frame_id = frame_id
            self._last_capture_ts_ns = capture_ts_ns
            self._published_batches += 1
            self._object_meta_frames += 1
            self._last_batch_age_ms = batch_age_ms
            self._publish_samples.append((publish_ts_ns, batch_age_ms))
            self._input_age_samples.append((publish_ts_ns, input_age_ms))
            self._inference_samples.append((publish_ts_ns, inference_ms))
            self._build_samples.append((publish_ts_ns, build_ms))
            self._last_error = ""

    def _object_meta_detections(self, pyds: Any, frame_meta: Any) -> list[Detection]:
        scale_x = self.pipeline_config.roi_width / self.pipeline_config.model_width
        scale_y = self.pipeline_config.roi_height / self.pipeline_config.model_height
        detections: list[Detection] = []
        object_list = frame_meta.obj_meta_list
        while object_list is not None:
            object_meta = pyds.NvDsObjectMeta.cast(object_list.data)
            rect = object_meta.rect_params
            left = float(rect.left) * scale_x
            top = float(rect.top) * scale_y
            right = (float(rect.left) + float(rect.width)) * scale_x
            bottom = (float(rect.top) + float(rect.height)) * scale_y
            left = max(0.0, min(float(self.pipeline_config.roi_width), left))
            top = max(0.0, min(float(self.pipeline_config.roi_height), top))
            right = max(0.0, min(float(self.pipeline_config.roi_width), right))
            bottom = max(0.0, min(float(self.pipeline_config.roi_height), bottom))
            confidence = float(getattr(object_meta, "confidence", 0.0))
            if right > left and bottom > top and 0.0 <= confidence <= 1.0:
                detections.append(
                    Detection(
                        cls=int(object_meta.class_id),
                        score=confidence,
                        box=BBox.from_xyxy(left, top, right, bottom),
                    )
                )
            object_list = object_list.next
        return detections

    def _capture_ts_from_pts(self, pts_ns: int, *, observed_ns: int) -> int:
        pts_ns = int(pts_ns)
        observed_ns = int(observed_ns)
        if pts_ns < 0 or pts_ns >= GST_CLOCK_TIME_NONE:
            with self._lock:
                self._timestamp_source = "observed_probe_time_invalid_pts"
            return observed_ns
        mapped = self._capture_ts_from_pipeline_clock(pts_ns, observed_ns=observed_ns)
        if mapped is not None:
            with self._lock:
                self._timestamp_source = "gst_clock_base_time_pts"
            return mapped
        with self._lock:
            if self._fallback_pts_offset_ns is None:
                self._fallback_pts_offset_ns = observed_ns - pts_ns
            self._timestamp_source = "first_probe_offset_pts"
            return pts_ns + self._fallback_pts_offset_ns

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

    def _start_bus_monitor(self, pipeline: Any) -> None:
        self._bus_stop.clear()
        thread = threading.Thread(
            target=self._bus_monitor_loop,
            args=(pipeline,),
            name="novasight-deepstream-bus",
            daemon=True,
        )
        self._bus_thread = thread
        thread.start()

    def _stop_bus_monitor(self) -> None:
        self._bus_stop.set()
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
                if message.type == Gst.MessageType.ERROR:
                    error, debug = message.parse_error()
                    self._record_terminal_error(f"DeepStream pipeline error: {error}: {debug}")
                else:
                    self._record_terminal_error("DeepStream pipeline reached EOS")
                return
        except Exception as exc:
            self._record_terminal_error(f"DeepStream bus monitor failed: {exc}")

    def _record_terminal_error(self, reason: str) -> None:
        with self._lock:
            self._last_error = reason
            self._terminal_error = True
            self._running = False
        self._bus_stop.set()

    def _prune_samples_locked(self, now_ns: int) -> None:
        threshold = int(now_ns) - 60_000_000_000
        for samples in (
            self._input_frame_samples,
            self._publish_samples,
            self._input_age_samples,
            self._inference_samples,
            self._build_samples,
        ):
            while samples and samples[0][0] < threshold:
                samples.popleft()


def _check_dependencies(parser_library_path: Path) -> DeepStreamDependencyStatus:
    if not parser_library_path.is_file():
        return DeepStreamDependencyStatus(
            False,
            "deepstream-parser-library-missing",
            str(parser_library_path),
        )
    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        Gst = importlib.import_module("gi.repository.Gst")
        Gst.init(None)
    except Exception as exc:
        return DeepStreamDependencyStatus(False, "gstreamer-python-unavailable", str(exc))
    try:
        importlib.import_module("pyds")
    except Exception as exc:
        return DeepStreamDependencyStatus(False, "pyds-unavailable", str(exc))
    missing = []
    find = getattr(getattr(Gst, "ElementFactory", None), "find", None)
    if callable(find):
        for name in ("nvv4l2decoder", "nvvidconv", "nvstreammux", "nvinfer"):
            if find(name) is None:
                missing.append(name)
    if missing:
        return DeepStreamDependencyStatus(
            False,
            "deepstream-gst-elements-unavailable",
            "missing GStreamer elements: " + ", ".join(missing),
        )
    return DeepStreamDependencyStatus(True)


def check_deepstream_dependencies(
    parser_library_path: Path | str = "build/deepstream-parser/libnovasight_parser.so",
) -> DeepStreamDependencyStatus:
    return _check_dependencies(Path(parser_library_path).expanduser().resolve(strict=False))


def _buffer_pts_ns(buffer: Any) -> int:
    try:
        return int(getattr(buffer, "pts"))
    except (AttributeError, TypeError, ValueError):
        return GST_CLOCK_TIME_NONE


def _sample_stats(samples: deque[tuple[int, float]]) -> dict[str, float | int]:
    values = sorted(float(value) for _timestamp, value in samples)
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": values[-1],
    }


def _sample_rate(samples: deque[tuple[int, float]], now_ns: int) -> float:
    threshold_ns = int(now_ns) - 1_000_000_000
    count = sum(1 for timestamp_ns, _value in samples if timestamp_ns >= threshold_ns)
    return float(count)


def _percentile(values: list[float], fraction: float) -> float:
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return values[index]


def _set_pipeline_null_best_effort(Gst: Any, pipeline: Any) -> None:
    try:
        pipeline.set_state(Gst.State.NULL)
    except Exception:
        pass


__all__ = [
    "DeepStreamDependencyStatus",
    "DeepStreamObjectBackend",
    "check_deepstream_dependencies",
]
