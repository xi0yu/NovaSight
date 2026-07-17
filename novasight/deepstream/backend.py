from __future__ import annotations

import ctypes
import importlib
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from novasight.contracts import BBox, Detection, DetectionBatch
from novasight.latency import pipeline_latency_breakdown_ms
from novasight.model_registry.manifest import ModelManifest
from novasight.detection_batch_mailbox import DetectionBatchMailbox

from .pipeline_builder import DeepStreamPipelineConfig, build_deepstream_pipeline
from .parser_build import ensure_deepstream_parser_library
from .parser_presets import resolve_parser_plan


GST_CLOCK_TIME_NONE = (1 << 64) - 1
PARSER_TELEMETRY_INTERVAL_NS = 100_000_000
TRUSTED_TIMESTAMP_SOURCES = {
    "gst_clock_base_time_pts",
    "first_probe_offset_pts",
}
logger = logging.getLogger("novasight.deepstream.backend")


@dataclass(frozen=True, slots=True)
class DeepStreamDependencyStatus:
    available: bool
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _InferenceInputTiming:
    pts_ns: int
    capture_ts_ns: int
    inference_start_ts_ns: int
    timestamp_source: str


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
        self._preview_requested = bool(pipeline_config.preview_enabled)
        self._crosshair_requested = bool(pipeline_config.crosshair_enabled)
        self._preview_active = False
        self._preview_consumers = 0
        self._preview_disabled_reason = ""
        self._preview_negotiation_fallback_attempted = False
        self.manifest = manifest
        self._parser_plan = resolve_parser_plan(
            manifest.postprocess.parser_preset,
            output_shape=manifest.output.shape,
            class_count=manifest.output.class_count,
            inferred_has_objectness=manifest.output.has_objectness,
        )
        self.parser_library_path = Path(parser_library_path).expanduser().resolve(strict=False)
        self.max_publish_age_ms = max(0.0, float(max_publish_age_ms))
        self.pipeline_description = build_deepstream_pipeline(pipeline_config)
        self.detection_batch_mailbox = DetectionBatchMailbox()
        self._lock = threading.RLock()
        self._preview_condition = threading.Condition(self._lock)
        self._crosshair_condition = threading.Condition(self._lock)
        self._pipeline: Any | None = None
        self._gst_module: Any | None = None
        self._pyds_module: Any | None = None
        self._running = False
        self._terminal_error = False
        self._last_error = ""
        self._bus_stop = threading.Event()
        self._first_batch_event = threading.Event()
        self._bus_thread: threading.Thread | None = None
        self._dependency_status: DeepStreamDependencyStatus | None = None
        self._parser_telemetry = _ParserTelemetry(self.parser_library_path)
        self._parser_status_lock = threading.Lock()
        self._cached_parser_status: dict[str, object] = {}
        self._cached_parser_status_ts_ns = 0
        self._parser_auto_build: dict[str, object] = {
            "attempted": False,
            "success": self.parser_library_path.is_file(),
            "detail": (
                "existing library" if self.parser_library_path.is_file() else "not attempted"
            ),
        }
        self._inference_start_by_pts: OrderedDict[
            int, _InferenceInputTiming | int
        ] = OrderedDict()
        self._gst_to_monotonic_offset_ns: int | None = None
        self._gst_base_time_ns: int | None = None
        self._fallback_pts_offset_ns: int | None = None
        self._timestamp_source = "uninitialized"
        self._generation = -1
        self._last_frame_id = -1
        self._last_capture_ts_ns = 0
        self._last_capture_interval_ms = 0.0
        self._capture_frames = 0
        self._input_frames = 0
        self._output_buffers = 0
        self._batch_meta_buffers = 0
        self._frame_meta_frames = 0
        self._published_batches = 0
        self._stale_dropped_batches = 0
        self._non_monotonic_dropped_batches = 0
        self._timestamp_rejected_batches = 0
        self._timestamp_buffer_pts_matches = 0
        self._timestamp_frame_meta_pts_matches = 0
        self._timestamp_correlation_misses = 0
        self._object_meta_frames = 0
        self._preview_frames = 0
        self._preview_sequence = 0
        self._latest_preview_jpeg: bytes | None = None
        self._last_preview_ts_ns = 0
        self._preview_error = ""
        self._crosshair_frames = 0
        self._crosshair_sequence = 0
        self._latest_crosshair_jpeg: bytes | None = None
        self._last_crosshair_ts_ns = 0
        self._crosshair_error = ""
        self._last_batch_age_ms = 0.0
        self._capture_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._input_frame_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._output_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._publish_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._preview_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._input_age_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._inference_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._build_samples: deque[tuple[int, float]] = deque(maxlen=16_384)
        self._started_at_ns = 0
        self._last_progress_log_ns = 0

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

    @property
    def preview_active(self) -> bool:
        with self._lock:
            return (
                self._preview_active
                and self._running
                and not self._terminal_error
                and bool(self.pipeline_config.preview_enabled)
            )

    def set_preview_active(self, enabled: bool) -> dict[str, object]:
        requested = bool(enabled)
        with self._lock:
            if not self.pipeline_config.preview_enabled:
                raise RuntimeError(
                    self._preview_disabled_reason or "hardware preview is unavailable"
                )
            pipeline = self._pipeline
            if pipeline is None or not self._running or self._terminal_error:
                raise RuntimeError("DeepStream pipeline is not running")
            valve = pipeline.get_by_name("preview-valve")
            if valve is None:
                raise RuntimeError("DeepStream pipeline missing preview-valve")

        valve.set_property("drop", not requested)
        with self._preview_condition:
            self._preview_active = requested
            if not requested:
                self._preview_consumers = 0
                self._latest_preview_jpeg = None
                self._last_preview_ts_ns = 0
                self._preview_error = ""
            self._preview_condition.notify_all()
        logger.info("DeepStream hardware preview active=%s", requested)
        return self.status()

    def acquire_preview_consumer(self) -> None:
        with self._lock:
            if not self._preview_active:
                raise RuntimeError("DeepStream hardware preview is paused")
            self._preview_consumers += 1

    def release_preview_consumer(self) -> None:
        with self._lock:
            if self._preview_consumers <= 0:
                return
            self._preview_consumers -= 1
            should_disable = self._preview_consumers == 0 and self._preview_active
        if not should_disable:
            return
        try:
            self.set_preview_active(False)
        except RuntimeError:
            # Pipeline shutdown may race with StreamingResponse cleanup.
            with self._preview_condition:
                self._preview_active = False
                self._latest_preview_jpeg = None
                self._last_preview_ts_ns = 0
                self._preview_condition.notify_all()

    def wait_until_ready(self, timeout_s: float) -> bool:
        """Wait until this pipeline has published its first valid DetectionBatch."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            if not self.running:
                return False
            if self._first_batch_event.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._first_batch_event.wait(timeout=min(0.05, remaining))

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
        self._gst_module = Gst
        self._pyds_module = importlib.import_module("pyds")
        Gst.init(None)
        self._degrade_preview_if_unavailable(Gst)
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
                detail = _take_pipeline_error(Gst, pipeline, timeout_ns=500_000_000)
                raise RuntimeError(
                    detail or "failed to set DeepStream pipeline to PLAYING"
                )
            logger.info(
                "DeepStream pipeline start accepted state_change=%s model=%s output=%s "
                "classes=%s objectness=%s preview=%s crosshair=%s "
                "capture=%s:%sx%s@%s format=%s io_mode=%s",
                result,
                self.manifest.model_id,
                self.manifest.output.shape,
                self.manifest.output.class_count,
                self.manifest.output.has_objectness,
                self.pipeline_config.preview_enabled,
                self.pipeline_config.crosshair_enabled,
                self.pipeline_config.device,
                self.pipeline_config.capture_width,
                self.pipeline_config.capture_height,
                self.pipeline_config.fps,
                self.pipeline_config.pixel_format,
                self.pipeline_config.io_mode,
            )
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
            self._first_batch_event.clear()
            self.detection_batch_mailbox.clear()
            self._preview_condition.notify_all()
            self._crosshair_condition.notify_all()
        if pipeline is None:
            return
        try:
            Gst = importlib.import_module("gi.repository.Gst")
            detail = _set_pipeline_null_best_effort(Gst, pipeline)
            if detail:
                logger.warning("DeepStream pipeline stop incomplete: %s", detail)
                with self._lock:
                    self._last_error = detail
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)

    def status(self) -> dict[str, object]:
        dependency = self.dependency_status()
        now_ns = time.monotonic_ns()
        with self._lock:
            self._prune_samples_locked(now_ns)
            # Pad probes append to these queues while holding the same lock.
            # Snapshot them here, then sort the snapshots after releasing the
            # lock so telemetry cannot stall the DeepStream streaming thread.
            publish_samples = tuple(self._publish_samples)
            input_age_samples = tuple(self._input_age_samples)
            inference_samples = tuple(self._inference_samples)
            build_samples = tuple(self._build_samples)
            running = self._running and not self._terminal_error
            uptime_ms = (
                max(0.0, (now_ns - self._started_at_ns) / 1e6)
                if self._started_at_ns
                else 0.0
            )
            payload: dict[str, object] = {
                "selected": self.backend_id,
                "available": dependency.available,
                "loaded": running,
                "configured": True,
                "reason": dependency.reason,
                "detail": dependency.detail,
                "running": running,
                "terminal_error": self._terminal_error,
                "last_error": self._last_error,
                "pipeline": self.pipeline_description,
                "timestamp_source": self._timestamp_source,
                "clock_domain": "monotonic",
                "capture_frames": self._capture_frames,
                "capture_fps": _sample_rate(self._capture_samples, now_ns),
                "input_frames": self._input_frames,
                "input_fps": _sample_rate(self._input_frame_samples, now_ns),
                "output_buffers": self._output_buffers,
                "output_fps": _sample_rate(self._output_samples, now_ns),
                "batch_meta_buffers": self._batch_meta_buffers,
                "frame_meta_frames": self._frame_meta_frames,
                "published_batches": self._published_batches,
                "published_fps": _sample_rate(self._publish_samples, now_ns),
                "stale_dropped_batches": self._stale_dropped_batches,
                "max_publish_age_ms": self.max_publish_age_ms,
                "non_monotonic_dropped_batches": self._non_monotonic_dropped_batches,
                "timestamp_rejected_batches": self._timestamp_rejected_batches,
                "timestamp_buffer_pts_matches": self._timestamp_buffer_pts_matches,
                "timestamp_frame_meta_pts_matches": self._timestamp_frame_meta_pts_matches,
                "timestamp_correlation_misses": self._timestamp_correlation_misses,
                "object_meta_frames": self._object_meta_frames,
                "preview_enabled": bool(self.pipeline_config.preview_enabled),
                "preview_active": self._preview_active,
                "preview_requested": self._preview_requested,
                "preview_negotiation_fallback_attempted": (
                    self._preview_negotiation_fallback_attempted
                ),
                "preview_available": self._latest_preview_jpeg is not None,
                "preview_frames": self._preview_frames,
                "preview_fps": _sample_rate(self._preview_samples, now_ns),
                "preview_sequence": self._preview_sequence,
                "preview_last_age_ms": (
                    max(0.0, (now_ns - self._last_preview_ts_ns) / 1e6)
                    if self._last_preview_ts_ns > 0
                    else 0.0
                ),
                "preview_reason": self._preview_reason_locked(),
                "preview_transport": "nvmm_nvjpegenc_to_mjpeg_bytes",
                "crosshair_requested": self._crosshair_requested,
                "crosshair_enabled": bool(self.pipeline_config.crosshair_enabled),
                "crosshair_active": bool(
                    self.pipeline_config.crosshair_enabled and self._running
                ),
                "crosshair_available": self._latest_crosshair_jpeg is not None,
                "crosshair_frames": self._crosshair_frames,
                "crosshair_sequence": self._crosshair_sequence,
                "crosshair_last_age_ms": (
                    max(0.0, (now_ns - self._last_crosshair_ts_ns) / 1e6)
                    if self._last_crosshair_ts_ns > 0
                    else 0.0
                ),
                "crosshair_error": self._crosshair_error,
                "crosshair_reason": self._crosshair_reason_locked(),
                "crosshair_transport": "nvmm_center_crop_nvjpeg_bytes",
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
                "input_name": self.manifest.input.name,
                "input_shape": "x".join(str(value) for value in self.manifest.input.shape),
                "input_dtype": self.manifest.input.dtype,
                "input_layout": self.manifest.input.layout,
                "runtime_precision": self.manifest.runtime.precision,
                "output_name": self.manifest.output.name,
                "output_shape": "x".join(str(value) for value in self.manifest.output.shape),
                "output_dtype": self.manifest.output.dtype,
                "model_output": {
                    "name": self.manifest.output.name,
                    "shape": list(self.manifest.output.shape),
                    "class_count": int(self.manifest.output.class_count),
                    "class_names": list(self.manifest.output.class_names),
                    "has_objectness": bool(self.manifest.output.has_objectness),
                },
                "postprocess": {
                    "parser": self.manifest.postprocess.parser,
                    "parser_preset": self.manifest.postprocess.parser_preset,
                    "compatibility": self._parser_plan.compatibility,
                    "has_objectness": self._parser_plan.has_objectness,
                    "parser_library": self._parser_plan.parser_library,
                    "parser_function": self._parser_plan.parser_function,
                    "nms_owner": self._parser_plan.nms_owner,
                    "confidence_threshold": self.manifest.postprocess.confidence_threshold,
                    "nms_threshold": self.manifest.postprocess.nms_iou_threshold,
                },
                "model_fingerprint": self.manifest.model_fingerprint,
                "output_sync_copy_ms": None,
                "output_sync_copy_reason": "nvinfer does not expose per-frame copy timing",
                "nms_ms": None,
                "nms_timing_reason": "DeepStream cluster-mode=2 has no per-frame NMS timing API",
                "parser": self._parser_status_snapshot(now_ns=now_ns, force=True),
                "detection_batch_mailbox": self.detection_batch_mailbox.status(),
                "uptime_ms": uptime_ms,
            }
            phase, phase_reason = self._inference_phase_locked(payload["parser"])
            payload["inference_phase"] = phase
            payload["inference_reason"] = phase_reason
        payload["batch_age_ms_stats"] = _sample_stats(publish_samples)
        payload["inference_input_age_ms_stats"] = _sample_stats(input_age_samples)
        payload["nvinfer_stage_ms_stats"] = _sample_stats(inference_samples)
        payload["nvinfer_timing_scope"] = "sink_to_src_including_parser"
        payload["detection_batch_build_ms_stats"] = _sample_stats(build_samples)
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

    def _degrade_preview_if_unavailable(self, Gst: Any) -> None:
        if not self._preview_requested and not self.pipeline_config.crosshair_enabled:
            return
        if self._preview_negotiation_fallback_attempted:
            if self.pipeline_config.preview_enabled or self.pipeline_config.crosshair_enabled:
                self.pipeline_config = replace(
                    self.pipeline_config,
                    preview_enabled=False,
                    crosshair_enabled=False,
                )
                self.pipeline_description = build_deepstream_pipeline(self.pipeline_config)
            self._preview_active = False
            return
        find = getattr(getattr(Gst, "ElementFactory", None), "find", None)
        missing = [
            name
            for name in ("nvjpegenc", "videorate", "appsink")
            if callable(find) and find(name) is None
        ]
        if not missing:
            if self._preview_requested and not self.pipeline_config.preview_enabled:
                self.pipeline_config = replace(self.pipeline_config, preview_enabled=True)
                self.pipeline_description = build_deepstream_pipeline(self.pipeline_config)
            self._preview_disabled_reason = ""
            return
        self._preview_disabled_reason = (
            "hardware JPEG branches disabled; missing GStreamer element(s): "
            + ", ".join(missing)
        )
        self.pipeline_config = replace(
            self.pipeline_config,
            preview_enabled=False,
            crosshair_enabled=False,
        )
        self._preview_active = False
        self._preview_consumers = 0
        self.pipeline_description = build_deepstream_pipeline(self.pipeline_config)
        logger.warning(self._preview_disabled_reason)

    def _reset_state_locked(self) -> None:
        self.detection_batch_mailbox.clear()
        self._first_batch_event.clear()
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
        self._capture_frames = 0
        self._input_frames = 0
        self._output_buffers = 0
        self._batch_meta_buffers = 0
        self._frame_meta_frames = 0
        self._published_batches = 0
        self._stale_dropped_batches = 0
        self._non_monotonic_dropped_batches = 0
        self._timestamp_rejected_batches = 0
        self._timestamp_buffer_pts_matches = 0
        self._timestamp_frame_meta_pts_matches = 0
        self._timestamp_correlation_misses = 0
        self._object_meta_frames = 0
        self._preview_frames = 0
        self._preview_active = False
        self._preview_sequence = 0
        self._latest_preview_jpeg = None
        self._last_preview_ts_ns = 0
        self._preview_error = ""
        self._crosshair_frames = 0
        self._crosshair_sequence = 0
        self._latest_crosshair_jpeg = None
        self._last_crosshair_ts_ns = 0
        self._crosshair_error = ""
        self._last_batch_age_ms = 0.0
        self._capture_samples.clear()
        self._input_frame_samples.clear()
        self._output_samples.clear()
        self._publish_samples.clear()
        self._preview_samples.clear()
        self._input_age_samples.clear()
        self._inference_samples.clear()
        self._build_samples.clear()
        self._cached_parser_status = {}
        self._cached_parser_status_ts_ns = 0
        self._last_progress_log_ns = 0

    def _attach_probes(self, Gst: Any, pipeline: Any) -> None:
        capture_source = pipeline.get_by_name("capture-source")
        nvinfer = pipeline.get_by_name("primary-infer")
        if capture_source is None:
            raise RuntimeError("DeepStream pipeline missing capture-source")
        if nvinfer is None:
            raise RuntimeError("DeepStream pipeline missing primary-infer")
        capture_pad = capture_source.get_static_pad("src")
        sink_pad = nvinfer.get_static_pad("sink")
        src_pad = nvinfer.get_static_pad("src")
        if capture_pad is None:
            raise RuntimeError("DeepStream capture-source src pad is unavailable")
        if sink_pad is None or src_pad is None:
            raise RuntimeError("DeepStream primary-infer pads are unavailable")
        capture_pad.add_probe(Gst.PadProbeType.BUFFER, self._capture_probe)
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._inference_start_probe)
        src_pad.add_probe(Gst.PadProbeType.BUFFER, self._object_meta_probe)
        if self.pipeline_config.preview_enabled:
            preview_valve = pipeline.get_by_name("preview-valve")
            preview_sink = pipeline.get_by_name("preview_sink")
            if preview_valve is None:
                raise RuntimeError("DeepStream pipeline missing preview-valve")
            if preview_sink is None:
                raise RuntimeError("DeepStream pipeline missing preview_sink")
            preview_pad = preview_sink.get_static_pad("sink")
            if preview_pad is None:
                raise RuntimeError("DeepStream preview sink pad is unavailable")
            preview_pad.add_probe(Gst.PadProbeType.BUFFER, self._preview_jpeg_probe)
        if self.pipeline_config.crosshair_enabled:
            crosshair_sink = pipeline.get_by_name("crosshair_sink")
            if crosshair_sink is None:
                raise RuntimeError("DeepStream pipeline missing crosshair_sink")
            crosshair_pad = crosshair_sink.get_static_pad("sink")
            if crosshair_pad is None:
                raise RuntimeError("DeepStream crosshair sink pad is unavailable")
            crosshair_pad.add_probe(Gst.PadProbeType.BUFFER, self._crosshair_jpeg_probe)
        logger.info("DeepStream capture and nvinfer sink/src probes attached")

    def _capture_probe(self, _pad: Any, info: Any) -> Any:
        Gst = self._gst_module or importlib.import_module("gi.repository.Gst")
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        now_ns = time.monotonic_ns()
        with self._lock:
            if self._terminal_error or not self._running:
                return Gst.PadProbeReturn.DROP
            self._capture_frames += 1
            self._capture_samples.append((now_ns, 0.0))
        return Gst.PadProbeReturn.OK

    def _inference_start_probe(self, _pad: Any, info: Any) -> Any:
        Gst = self._gst_module or importlib.import_module("gi.repository.Gst")
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
                self._inference_start_by_pts[pts_ns] = _InferenceInputTiming(
                    pts_ns=pts_ns,
                    capture_ts_ns=capture_ts_ns,
                    inference_start_ts_ns=now_ns,
                    timestamp_source=self._timestamp_source,
                )
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
        Gst = self._gst_module or importlib.import_module("gi.repository.Gst")
        try:
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            now_ns = time.monotonic_ns()
            with self._lock:
                self._output_buffers += 1
                self._output_samples.append((now_ns, 0.0))
            pyds = self._pyds_module or importlib.import_module("pyds")
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch_meta is None:
                raise RuntimeError("nvinfer output buffer has no NvDsBatchMeta")
            with self._lock:
                self._batch_meta_buffers += 1
            frame_list = batch_meta.frame_meta_list
            while frame_list is not None:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                with self._lock:
                    self._frame_meta_frames += 1
                self._publish_frame_meta(pyds, frame_meta, buffer)
                frame_list = _next_meta_node(frame_list)
        except Exception as exc:
            self._record_terminal_error(f"DeepStream object-meta probe failed: {exc}")
        return Gst.PadProbeReturn.OK

    def _preview_jpeg_probe(self, _pad: Any, info: Any) -> Any:
        Gst = self._gst_module or importlib.import_module("gi.repository.Gst")
        with self._lock:
            if not self._preview_active:
                return Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        mapped = False
        map_info = None
        try:
            mapped, map_info = buffer.map(Gst.MapFlags.READ)
            if not mapped or map_info is None:
                raise RuntimeError("hardware JPEG buffer could not be mapped")
            payload = bytes(map_info.data)
            if len(payload) < 4 or not payload.startswith(b"\xff\xd8"):
                raise RuntimeError("hardware preview buffer is not a JPEG image")
            now_ns = time.monotonic_ns()
            with self._preview_condition:
                self._preview_frames += 1
                self._preview_sequence += 1
                self._latest_preview_jpeg = payload
                self._last_preview_ts_ns = now_ns
                self._preview_error = ""
                self._preview_samples.append((now_ns, 0.0))
                self._preview_condition.notify_all()
        except Exception as exc:
            with self._lock:
                self._preview_error = str(exc)
            logger.warning("DeepStream hardware preview frame rejected: %s", exc)
        finally:
            if mapped and map_info is not None:
                buffer.unmap(map_info)
        return Gst.PadProbeReturn.OK

    def _crosshair_jpeg_probe(self, _pad: Any, info: Any) -> Any:
        Gst = self._gst_module or importlib.import_module("gi.repository.Gst")
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        mapped = False
        map_info = None
        try:
            mapped, map_info = buffer.map(Gst.MapFlags.READ)
            if not mapped or map_info is None:
                raise RuntimeError("crosshair JPEG buffer could not be mapped")
            payload = bytes(map_info.data)
            if len(payload) < 4 or not payload.startswith(b"\xff\xd8"):
                raise RuntimeError("crosshair buffer is not a JPEG image")
            now_ns = time.monotonic_ns()
            with self._crosshair_condition:
                self._crosshair_frames += 1
                self._crosshair_sequence += 1
                self._latest_crosshair_jpeg = payload
                self._last_crosshair_ts_ns = now_ns
                self._crosshair_error = ""
                self._crosshair_condition.notify_all()
        except Exception as exc:
            with self._lock:
                self._crosshair_error = str(exc)
            logger.warning("DeepStream crosshair frame rejected: %s", exc)
        finally:
            if mapped and map_info is not None:
                buffer.unmap(map_info)
        return Gst.PadProbeReturn.OK

    def wait_preview_jpeg(
        self,
        *,
        after_sequence: int | None = None,
        timeout_s: float = 0.0,
    ) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._preview_condition:
            while True:
                if not self._preview_active:
                    return None
                if (
                    self._latest_preview_jpeg is not None
                    and (after_sequence is None or self._preview_sequence > int(after_sequence))
                ):
                    return self._preview_sequence, self._latest_preview_jpeg
                if not self._running or self._terminal_error:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._preview_condition.wait(timeout=remaining)

    def wait_crosshair_jpeg(
        self,
        *,
        after_sequence: int | None = None,
        timeout_s: float = 0.0,
    ) -> tuple[int, bytes, int] | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._crosshair_condition:
            while True:
                if not self.pipeline_config.crosshair_enabled:
                    return None
                if (
                    self._latest_crosshair_jpeg is not None
                    and (
                        after_sequence is None
                        or self._crosshair_sequence > int(after_sequence)
                    )
                ):
                    return (
                        self._crosshair_sequence,
                        self._latest_crosshair_jpeg,
                        self._last_crosshair_ts_ns,
                    )
                if not self.running:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._crosshair_condition.wait(timeout=remaining)

    def _publish_frame_meta(self, pyds: Any, frame_meta: Any, buffer: Any) -> None:
        inference_end_ts_ns = time.monotonic_ns()
        buffer_pts_ns = _buffer_pts_ns(buffer)
        frame_meta_pts_value = getattr(frame_meta, "buf_pts", None)
        frame_meta_pts_ns = int(
            GST_CLOCK_TIME_NONE if frame_meta_pts_value is None else frame_meta_pts_value
        )
        timing, timestamp_correlation = self._take_input_timing(
            buffer_pts_ns=buffer_pts_ns,
            frame_meta_pts_ns=frame_meta_pts_ns,
            observed_ns=inference_end_ts_ns,
        )
        if timing is None:
            with self._lock:
                self._timestamp_rejected_batches += 1
                self._last_error = (
                    "DeepStream output could not be correlated with an nvinfer input: "
                    f"buffer_pts={buffer_pts_ns} frame_meta_pts={frame_meta_pts_ns}"
                )
            return
        raw_pts_ns = timing.pts_ns
        capture_ts_ns = timing.capture_ts_ns
        inference_start_ts_ns = timing.inference_start_ts_ns
        timestamp_source = timing.timestamp_source
        detections = self._object_meta_detections(pyds, frame_meta)
        publish_ts_ns = time.monotonic_ns()
        frame_id_value = getattr(frame_meta, "frame_num", None)
        frame_id = int(self._last_frame_id + 1 if frame_id_value is None else frame_id_value)
        parser = self._parser_status_snapshot(now_ns=publish_ts_ns)
        latency = pipeline_latency_breakdown_ms(
            capture_ts_ns=capture_ts_ns,
            inference_start_ts_ns=inference_start_ts_ns,
            inference_end_ts_ns=inference_end_ts_ns,
            publish_ts_ns=publish_ts_ns,
        )
        input_age_ms = latency["ingress_ms"]
        inference_ms = latency["inference_ms"]
        batch_age_ms = latency["publish_age_ms"]
        build_ms = latency["batch_build_ms"]
        with self._lock:
            if self._terminal_error or not self._running:
                return
            if timestamp_source not in TRUSTED_TIMESTAMP_SOURCES:
                self._timestamp_rejected_batches += 1
                self._last_error = (
                    "DeepStream timestamp does not have a stable monotonic mapping: "
                    f"{timestamp_source}"
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
                "timestamp_source": timestamp_source,
                "raw_pts_ns": raw_pts_ns,
                "output_buffer_pts_ns": buffer_pts_ns,
                "frame_meta_pts_ns": frame_meta_pts_ns,
                "timestamp_correlation": timestamp_correlation,
                "postprocess_owner": "nvinfer_custom_parser_and_cluster_mode_2",
                "python_nms": False,
                "parser": parser,
                "nvinfer_stage_ms": inference_ms,
                "nvinfer_timing_scope": "sink_to_src_including_parser",
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
            self._first_batch_event.set()
            self._object_meta_frames += 1
            self._last_batch_age_ms = batch_age_ms
            self._publish_samples.append((publish_ts_ns, batch_age_ms))
            self._input_age_samples.append((publish_ts_ns, input_age_ms))
            self._inference_samples.append((publish_ts_ns, inference_ms))
            self._build_samples.append((publish_ts_ns, build_ms))
            self._last_error = ""

    def _take_input_timing(
        self,
        *,
        buffer_pts_ns: int,
        frame_meta_pts_ns: int,
        observed_ns: int,
    ) -> tuple[_InferenceInputTiming | None, str]:
        with self._lock:
            candidates = (
                ("buffer_pts", int(buffer_pts_ns)),
                ("frame_meta_pts", int(frame_meta_pts_ns)),
            )
            for correlation, pts_ns in candidates:
                stored = self._inference_start_by_pts.pop(pts_ns, None)
                if stored is None:
                    continue
                timing = self._coerce_input_timing_locked(
                    stored,
                    pts_ns=pts_ns,
                    observed_ns=observed_ns,
                )
                if correlation == "buffer_pts":
                    self._timestamp_buffer_pts_matches += 1
                else:
                    self._timestamp_frame_meta_pts_matches += 1
                return timing, correlation
            self._timestamp_correlation_misses += 1
            return None, "missing"

    def _coerce_input_timing_locked(
        self,
        stored: _InferenceInputTiming | int,
        *,
        pts_ns: int,
        observed_ns: int,
    ) -> _InferenceInputTiming:
        if isinstance(stored, _InferenceInputTiming):
            return stored
        capture_ts_ns = self._capture_ts_from_pts(pts_ns, observed_ns=observed_ns)
        return _InferenceInputTiming(
            pts_ns=pts_ns,
            capture_ts_ns=capture_ts_ns,
            inference_start_ts_ns=int(stored),
            timestamp_source=self._timestamp_source,
        )

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
            object_list = _next_meta_node(object_list)
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
            Gst = self._gst_module or importlib.import_module("gi.repository.Gst")
            bus = pipeline.get_bus()
            if bus is None:
                return
            mask = Gst.MessageType.ERROR | Gst.MessageType.EOS
            while not self._bus_stop.is_set():
                message = bus.timed_pop_filtered(100_000_000, mask)
                if message is None:
                    self._log_progress_if_due()
                    continue
                if message.type == Gst.MessageType.ERROR:
                    error, debug = message.parse_error()
                    reason = f"DeepStream pipeline error: {error}: {debug}"
                    if self._retry_without_preview_after_negotiation_error(reason):
                        return
                    self._record_terminal_error(reason)
                else:
                    self._record_terminal_error("DeepStream pipeline reached EOS")
                return
        except Exception as exc:
            self._record_terminal_error(f"DeepStream bus monitor failed: {exc}")

    def _retry_without_preview_after_negotiation_error(self, reason: str) -> bool:
        normalized = str(reason).lower()
        with self._lock:
            should_retry = (
                (self.pipeline_config.preview_enabled or self.pipeline_config.crosshair_enabled)
                and not self._preview_negotiation_fallback_attempted
                and self._input_frames == 0
                and "not-negotiated" in normalized
            )
            if not should_retry:
                return False
            self._preview_negotiation_fallback_attempted = True
            self._preview_disabled_reason = (
                "optional NVMM JPEG branches disabled after caps negotiation failed"
            )
            self.pipeline_config = replace(
                self.pipeline_config,
                preview_enabled=False,
                crosshair_enabled=False,
            )
            self._preview_active = False
            self.pipeline_description = build_deepstream_pipeline(self.pipeline_config)
        logger.warning(
            "DeepStream optional JPEG branch negotiation failed before nvinfer input; "
            "retrying inference pipeline without preview/crosshair reason=%s",
            reason,
        )
        try:
            self.start()
        except Exception as exc:
            self._record_terminal_error(
                "DeepStream pipeline preview fallback failed: "
                f"original={reason}; fallback={exc}"
            )
        return True

    def _record_terminal_error(self, reason: str) -> None:
        with self._preview_condition:
            self._last_error = reason
            self._terminal_error = True
            self._running = False
            self._preview_condition.notify_all()
            self._crosshair_condition.notify_all()
        self._bus_stop.set()
        logger.error("DeepStream terminal error: %s", reason)

    def _log_progress_if_due(self) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            if now_ns - self._last_progress_log_ns < 2_000_000_000:
                return
            self._last_progress_log_ns = now_ns
            progress = {
                "input": self._input_frames,
                "capture": self._capture_frames,
                "capture_fps": _sample_rate(self._capture_samples, now_ns),
                "input_fps": _sample_rate(self._input_frame_samples, now_ns),
                "output": self._output_buffers,
                "output_fps": _sample_rate(self._output_samples, now_ns),
                "batch_meta": self._batch_meta_buffers,
                "frame_meta": self._frame_meta_frames,
                "published": self._published_batches,
                "published_fps": _sample_rate(self._publish_samples, now_ns),
                "preview": self._preview_frames,
                "preview_fps": _sample_rate(self._preview_samples, now_ns),
                "stale": self._stale_dropped_batches,
                "timestamp_rejected": self._timestamp_rejected_batches,
                "timestamp_buffer_pts": self._timestamp_buffer_pts_matches,
                "timestamp_frame_meta_pts": self._timestamp_frame_meta_pts_matches,
                "timestamp_correlation_miss": self._timestamp_correlation_misses,
                "non_monotonic": self._non_monotonic_dropped_batches,
                "timestamp_source": self._timestamp_source,
                "last_error": self._last_error,
            }
        progress["parser"] = self._parser_status_snapshot(now_ns=now_ns, force=True)
        logger.info("DeepStream inference progress %s", progress)

    def _parser_status_snapshot(
        self,
        *,
        now_ns: int,
        force: bool = False,
    ) -> dict[str, object]:
        with self._parser_status_lock:
            if (
                force
                or not self._cached_parser_status
                or int(now_ns) - self._cached_parser_status_ts_ns
                >= PARSER_TELEMETRY_INTERVAL_NS
            ):
                self._cached_parser_status = self._parser_telemetry.snapshot()
                self._cached_parser_status_ts_ns = int(now_ns)
            return self._cached_parser_status

    def _preview_reason_locked(self) -> str:
        if not self.pipeline_config.preview_enabled:
            return self._preview_disabled_reason or "preview consumer is disabled"
        if not self._preview_active:
            return "preview paused to preserve inference performance"
        if self._preview_error:
            return self._preview_error
        if self._latest_preview_jpeg is None:
            return "waiting for the first hardware JPEG preview frame"
        return ""

    def _crosshair_reason_locked(self) -> str:
        if not self._crosshair_requested:
            return "crosshair observer is disabled"
        if not self.pipeline_config.crosshair_enabled:
            return self._preview_disabled_reason or "crosshair JPEG branch is unavailable"
        if self._crosshair_error:
            return self._crosshair_error
        if self._latest_crosshair_jpeg is None:
            return "waiting for the first crosshair sample"
        return ""

    def _inference_phase_locked(self, parser: object) -> tuple[str, str]:
        parser_status = parser if isinstance(parser, dict) else {}
        parser_calls = int(parser_status.get("decode_calls") or 0)
        parse_failures = int(parser_status.get("parse_failures") or 0)
        if self._terminal_error:
            return "failed", self._last_error or "DeepStream pipeline terminated"
        if not self._running:
            return "stopped", self._last_error or "DeepStream pipeline is not running"
        if self._input_frames <= 0:
            return "waiting_input", "waiting for the first nvinfer input buffer"
        if self._published_batches > 0:
            return "publishing", "DetectionBatch is being published"
        if self._output_buffers <= 0:
            return "waiting_output", "nvinfer has input buffers but no src output buffer"
        if self._batch_meta_buffers <= 0:
            return "missing_batch_meta", "nvinfer output has no readable NvDsBatchMeta"
        if parser_calls <= 0:
            return "parser_not_called", "NvDsInferParseNovaSight has not been called"
        if parse_failures > 0 and self._published_batches <= 0:
            code = parser_status.get("last_error_code")
            return "parser_failed", f"native parser failed code={code}"
        if self._frame_meta_frames <= 0:
            return "missing_frame_meta", "NvDsBatchMeta contains no frame metadata"
        return "publish_blocked", self._last_error or "DetectionBatch publish is blocked"

    def _prune_samples_locked(self, now_ns: int) -> None:
        threshold = int(now_ns) - 60_000_000_000
        for samples in (
            self._capture_samples,
            self._input_frame_samples,
            self._output_samples,
            self._publish_samples,
            self._preview_samples,
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


def _next_meta_node(node: Any) -> Any | None:
    try:
        return node.next
    except StopIteration:
        return None


def _sample_stats(samples: Iterable[tuple[int, float]]) -> dict[str, float | int]:
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


def _set_pipeline_null_best_effort(Gst: Any, pipeline: Any) -> str:
    try:
        result = pipeline.set_state(Gst.State.NULL)
        if result == Gst.StateChangeReturn.FAILURE:
            return _take_pipeline_error(Gst, pipeline, timeout_ns=100_000_000) or (
                "failed to set DeepStream pipeline to NULL"
            )
        get_state = getattr(pipeline, "get_state", None)
        if not callable(get_state):
            return ""
        state_result, current, pending = get_state(3_000_000_000)
        if state_result == Gst.StateChangeReturn.FAILURE:
            return _take_pipeline_error(Gst, pipeline, timeout_ns=100_000_000) or (
                "DeepStream pipeline failed while waiting for NULL"
            )
        if current != Gst.State.NULL:
            return f"DeepStream pipeline did not reach NULL (current={current}, pending={pending})"
        return ""
    except Exception as exc:
        return f"DeepStream pipeline stop failed: {exc}"


def _take_pipeline_error(Gst: Any, pipeline: Any, *, timeout_ns: int) -> str:
    try:
        bus = pipeline.get_bus()
        if bus is None:
            return ""
        message = bus.timed_pop_filtered(timeout_ns, Gst.MessageType.ERROR)
        if message is None:
            return ""
        error, debug = message.parse_error()
        return f"DeepStream pipeline error: {error}: {debug}"
    except Exception:
        return ""


__all__ = [
    "DeepStreamDependencyStatus",
    "DeepStreamObjectBackend",
    "check_deepstream_dependencies",
]
