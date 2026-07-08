from __future__ import annotations

from dataclasses import asdict
import logging
import math
import threading
import time
from typing import Any

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.coordinates import CoordinateTransform
from novasight.control import (
    ExperimentalAnglePidStrategy,
)
from novasight.deepstream import check_deepstream_dependencies
from novasight.executors import ExecutorRegistry
from novasight.hardware import BoxInputState
from novasight.inference import InferenceResult
from novasight.model_registry import ModelRegistry
from novasight.contracts import BBox, ControlIntent, Detection, DetectionBatch, FrameContext, Track
from novasight.roi import center_roi_frame, center_roi_region

from .config_store import RuntimeConfigStore
from .aim import (
    AimPointConfig,
    AimPointGenerator,
    LatencyCompensationConfig,
    LatencyCompensator,
    estimated_state_from_debug,
)
from .state import RuntimeFrameResult, RuntimeState
from .candidates import aim_point
from .detection_batch import detection_batch_to_frame_context
from .recorder import build_control_frame_record
from .target_selector import RuntimeTargetSelector, TargetSelection

logger = logging.getLogger("novasight.runtime.service")


def _status_float(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if number == number else float(fallback)


def _status_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


class RuntimeService:
    def __init__(
        self,
        config: RuntimeConfig,
        models: ModelRegistry,
        executors: ExecutorRegistry,
        hardware: Any | None = None,
        capture: Any | None = None,
        inference: Any | None = None,
        recorder: Any | None = None,
    ) -> None:
        self.config = config
        self.models = models
        self.executors = executors
        self.hardware = hardware
        self.capture = capture
        self.inference = inference
        self.recorder = recorder
        self.running = False
        self.config_store = RuntimeConfigStore(config)
        self.pipeline = None
        self.fatal_error: dict | None = None
        self.last_frame_context: FrameContext | None = None
        self.last_target: dict[str, Any] | None = None
        self.last_control: dict[str, Any] | None = None
        self.last_execution: dict[str, Any] | None = None
        self.last_inference_reason = ""
        self.last_pipeline_timings: dict[str, float] = {}
        self.last_inference_status: dict[str, Any] = {
            "ran": False,
            "available": False,
            "reason": "推理尚未执行",
        }
        self._last_control_log_signature = ""
        self._last_control_log_s = 0.0
        self._last_box_input_log_signature = ""
        self._last_box_input_log_s = 0.0
        self._control_lock = threading.Lock()
        self._last_control_tick_ns = 0
        self.control_strategy = self._create_control_strategy(config)
        self.target_selector = RuntimeTargetSelector()
        self.aim_points = AimPointGenerator()
        self.latency_compensator = LatencyCompensator()
        self._runtime_calibration_signature = self._config_calibration_signature(config)
        self._external_sensitivity_fingerprint = ""
        self._external_sensitivity_source = ""
        self._external_sensitivity_ts_ns = 0
        self._last_runtime_reset_reason = ""
        self._log_production_control_chain("startup")

    def state(self) -> RuntimeState:
        capture_state = getattr(self, "capture", None)
        inference_state = getattr(self, "inference", None)
        active_model = self._active_model()
        capture_payload = asdict(capture_state.state) if capture_state is not None else {}
        statistics = dict(capture_payload.get("statistics", {}))
        pipeline = getattr(self, "pipeline", None)
        pipeline_stats = getattr(pipeline, "stats", None)
        if pipeline_stats is not None:
            statistics["inference_counter"] = getattr(pipeline_stats, "window_processed_frames", 0)
            statistics["skipped_counter"] = getattr(pipeline_stats, "skipped_frames", 0)
            statistics["inference_fps"] = getattr(pipeline_stats, "inference_fps", 0.0)
            statistics["detection_batch_fps"] = getattr(pipeline_stats, "detection_batch_fps", 0.0)
            statistics["control_observation_counter"] = getattr(
                pipeline_stats,
                "window_control_observations",
                0,
            )
            statistics["control_observation_fps"] = getattr(
                pipeline_stats,
                "control_observation_fps",
                0.0,
            )
            statistics["queue_latency"] = getattr(pipeline_stats, "queue_latency_ms", 0.0)
            statistics["inference_latency"] = getattr(pipeline_stats, "inference_latency_ms", 0.0)
            statistics["e2e_latency"] = getattr(pipeline_stats, "e2e_latency_ms", 0.0)
        source_statistics = self._detection_source_statistics(pipeline)
        for key in (
            "published_batches",
            "window_published_batches",
            "tensor_meta_frames",
            "postprocess_frames",
            "window_tensor_meta_frames",
            "window_postprocess_frames",
            "tensor_meta_fps",
            "postprocess_fps",
            "timestamp_source",
            "last_frame_age_ms",
            "last_inference_latency_ms",
            "last_" + "detection_count",
        ):
            if key in source_statistics:
                statistics[key] = source_statistics[key]
        if source_statistics:
            tensor_meta_fps = _status_float(source_statistics.get("tensor_meta_fps"), 0.0)
            published_batches = _status_int(source_statistics.get("published_batches"), 0)
            if tensor_meta_fps > 0.0 and _status_float(statistics.get("capture_fps"), 0.0) <= 0.0:
                statistics["capture_fps"] = tensor_meta_fps
            if published_batches > 0 and _status_int(statistics.get("capture_counter"), 0) <= 0:
                statistics["capture_counter"] = published_batches
        for key, value in self.last_pipeline_timings.items():
            statistics[f"stage_{key}"] = value
        if capture_payload:
            capture_payload["statistics"] = statistics
        return RuntimeState(
            running=self.running,
            source=self.config.source.default,
            active_model=active_model,
            executor=self.executors.status(),
            capture=capture_payload,
            statistics=statistics,
            inference=self._runtime_inference_status(inference_state, active_model),
            config=self.config_store.status(),
            pipeline=self.pipeline.status() if self.pipeline is not None else {},
            vision=self._vision_status(),
            fatal_error=self.fatal_error,
        )

    @staticmethod
    def _detection_source_statistics(pipeline: Any) -> dict[str, Any]:
        detection_source = getattr(pipeline, "detection_source", None)
        status_fn = getattr(detection_source, "status", None)
        if not callable(status_fn):
            return {}
        try:
            source_status = status_fn()
        except Exception:
            return {}
        return dict(source_status) if isinstance(source_status, dict) else {}

    def _runtime_inference_status(
        self,
        inference_state: Any,
        active_model: dict | None,
    ) -> dict[str, Any]:
        backend = str(getattr(self.config.inference, "backend", "")).lower()
        if backend != "deepstream":
            return (
                inference_state.status()
                if inference_state is not None
                else {"available": False}
            )
        source_status: dict[str, Any] = {}
        pipeline = getattr(self, "pipeline", None)
        detection_source = getattr(pipeline, "detection_source", None)
        status_fn = getattr(detection_source, "status", None)
        if callable(status_fn):
            try:
                raw_status = status_fn()
                if isinstance(raw_status, dict):
                    source_status = dict(raw_status)
            except Exception as exc:
                source_status = {
                    "available": False,
                    "running": False,
                    "reason": "DeepStream status failed",
                    "detail": str(exc),
                }
        source_status.pop("pipeline", None)
        dependency_status: Any | None = None
        if not source_status:
            try:
                dependency_status = check_deepstream_dependencies()
            except Exception as exc:
                dependency_status = {
                    "available": False,
                    "reason": "DeepStream dependency check failed",
                    "detail": str(exc),
                }
        inference_config = getattr(self.config, "inference", None)
        explicit_deepstream_paths = bool(
            str(getattr(inference_config, "deepstream_manifest_path", "") or "").strip()
            and str(getattr(inference_config, "deepstream_config_path", "") or "").strip()
        )
        configured = active_model is not None or explicit_deepstream_paths
        running = bool(source_status.get("running", False))
        dependency_available = (
            bool(getattr(dependency_status, "available", False))
            if dependency_status is not None and not isinstance(dependency_status, dict)
            else bool((dependency_status or {}).get("available", False))
        )
        available = bool(
            source_status.get(
                "available",
                dependency_available if dependency_status is not None else configured,
            )
        )
        reason = str(source_status.get("reason") or source_status.get("last_error") or "")
        detail = str(source_status.get("detail") or "")
        if dependency_status is not None and not available:
            dependency_reason = (
                str(getattr(dependency_status, "reason", ""))
                if not isinstance(dependency_status, dict)
                else str(dependency_status.get("reason") or "")
            )
            dependency_detail = (
                str(getattr(dependency_status, "detail", ""))
                if not isinstance(dependency_status, dict)
                else str(dependency_status.get("detail") or "")
            )
            reason = reason or dependency_reason
            detail = detail or dependency_detail
        if not configured:
            reason = reason or "DeepStream backend requires an active model deployment"
        elif not running:
            reason = reason or "DeepStream nvinfer loads when runtime starts"
        return {
            **source_status,
            "selected": "deepstream",
            "backend": "deepstream",
            "available": available,
            "loaded": running,
            "running": running,
            "configured": configured,
            "reason": reason,
            "detail": detail,
        }

    def update_config(self, config: RuntimeConfig) -> RuntimeConfig:
        new_calibration_signature = self._config_calibration_signature(config)
        calibration_changed = new_calibration_signature != self._runtime_calibration_signature
        self.config = config
        self.control_strategy = self._create_control_strategy(config)
        if calibration_changed:
            self._clear_pending_commands("CALIBRATION_PROFILE_CHANGED")
        self.executors.update_runtime_config(config)
        if calibration_changed:
            self._runtime_calibration_signature = new_calibration_signature
            self._reset_runtime_control_state("CALIBRATION_PROFILE_CHANGED")
        configure = getattr(self.inference, "configure", None)
        if callable(configure):
            configure(
                confidence_threshold=config.inference.confidence_threshold,
                nms_threshold=config.inference.nms_threshold,
            )
        self._log_production_control_chain(
            "calibration_profile_changed" if calibration_changed else "config_updated"
        )
        return self.config_store.replace(config)

    def update_external_sensitivity_fingerprint(
        self,
        fingerprint: str,
        *,
        source: str = "external",
    ) -> dict[str, Any]:
        observed = str(fingerprint).strip()
        if not observed:
            raise ValueError("sensitivity fingerprint must be non-empty")
        source_name = str(source or "external").strip() or "external"
        with self._control_lock:
            previous_state = self._calibration_fingerprint_status()["state"]
            self._external_sensitivity_fingerprint = observed
            self._external_sensitivity_source = source_name
            self._external_sensitivity_ts_ns = time.monotonic_ns()
            status = self._calibration_fingerprint_status()
            if status["state"] != previous_state:
                self._reset_runtime_control_state(
                    "CALIBRATION_FINGERPRINT_MISMATCH"
                    if status["state"] == "mismatch"
                    else "CALIBRATION_FINGERPRINT_UPDATED"
                )
        logger.info(
            "calibration sensitivity fingerprint observed source=%s expected=%s observed=%s state=%s",
            source_name,
            status["expected_fingerprint"],
            status["observed_fingerprint"],
            status["state"],
        )
        return status

    @staticmethod
    def _config_calibration_signature(config: RuntimeConfig) -> tuple[Any, ...]:
        calibration = config.calibration
        capture = config.capture
        roi = config.roi
        return (
            str(calibration.profile_id).strip(),
            int(calibration.profile_version),
            str(calibration.fov_semantics).strip(),
            float(calibration.fov_x_deg),
            float(calibration.counts_per_360_x),
            float(calibration.counts_per_360_y),
            float(calibration.axis_sign_x),
            float(calibration.axis_sign_y),
            str(calibration.game_sensitivity_fingerprint).strip(),
            str(calibration.projection_profile).strip(),
            int(capture.width),
            int(capture.height),
            int(roi.size),
            str(roi.mode).strip(),
            int(roi.offset_x),
            int(roi.offset_y),
        )

    def _reset_runtime_control_state(self, reason: str) -> None:
        self._last_runtime_reset_reason = reason
        self.target_selector.reset()
        self.aim_points.reset()
        self.latency_compensator = LatencyCompensator()
        self._reset_control_motion_state()
        self._clear_pending_commands(reason)
        self.last_frame_context = None
        self.last_target = None
        self.last_execution = None
        self._last_control_tick_ns = 0
        self.last_control = {
            "will_emit": False,
            "control_allowed": False,
            "global_state": "DISABLED",
            "runtime_reset": True,
            "runtime_reset_reason": reason,
            "selector_state": "reset",
            "selection_reason": reason,
            "calibration_status": self._calibration_fingerprint_status(),
            "candidate_filter": self._candidate_filter_payload({}),
        }

    def _log_production_control_chain(self, event: str) -> None:
        calibration = self.config.calibration
        selected_executor = getattr(self.executors, "selected", None) or self.config.control.output_mode
        logger.info(
            "production_control_chain event=%s chain=%s strategy=%s executor=%s trigger=%s "
            "calibration_profile_id=%s calibration_profile_version=%s fov_x_deg=%.3f "
            "counts_per_360_x=%.3f counts_per_360_y=%.3f axis_sign_x=%.0f axis_sign_y=%.0f",
            event,
            "CompensatedTarget(Control px)->AngularErrorMapper->AngularPDController->CommandScheduler->kmNet",
            self.config.control.strategy,
            selected_executor,
            self.config.control.trigger_mode,
            calibration.profile_id,
            calibration.profile_version,
            float(calibration.fov_x_deg),
            float(calibration.counts_per_360_x),
            float(calibration.counts_per_360_y),
            float(calibration.axis_sign_x),
            float(calibration.axis_sign_y),
        )

    def _calibration_fingerprint_status(self) -> dict[str, Any]:
        expected = str(self.config.calibration.game_sensitivity_fingerprint).strip()
        observed = str(self._external_sensitivity_fingerprint).strip()
        source = str(self._external_sensitivity_source).strip()
        updated_ts_ns = int(self._external_sensitivity_ts_ns or 0)
        if not observed:
            return {
                "state": "unverified",
                "control_allowed": True,
                "expected_fingerprint": expected,
                "observed_fingerprint": "",
                "source": "",
                "updated_ts_ns": 0,
                "reason_code": "",
                "reason": "external sensitivity fingerprint has not been reported",
            }
        if observed == expected:
            return {
                "state": "matched",
                "control_allowed": True,
                "expected_fingerprint": expected,
                "observed_fingerprint": observed,
                "source": source,
                "updated_ts_ns": updated_ts_ns,
                "reason_code": "",
                "reason": "external sensitivity fingerprint matches calibration profile",
            }
        return {
            "state": "mismatch",
            "control_allowed": False,
            "expected_fingerprint": expected,
            "observed_fingerprint": observed,
            "source": source,
            "updated_ts_ns": updated_ts_ns,
            "reason_code": "CALIBRATION_FINGERPRINT_MISMATCH",
            "reason": "external sensitivity fingerprint does not match calibration profile",
        }

    def record_fatal_error(self, thread_name: str, exc: BaseException, path) -> None:
        self.fatal_error = {
            "type": "FATAL_ERROR",
            "thread": thread_name,
            "message": str(exc),
            "crash_log": str(path),
        }

    def process_frame(self, context: FrameContext) -> RuntimeFrameResult:
        with self._control_lock:
            self.last_frame_context = context
            intent = self._control_intent_from_context(context)
            self._last_control_tick_ns = time.monotonic_ns()
        control_intents = [intent] if intent is not None else []
        execution_results = [self.executors.execute(intent) for intent in control_intents]
        if execution_results:
            self.last_execution = self._execution_result_payload(execution_results[-1])
            self._attach_execution_to_last_control(self.last_execution)
            logger.info(
                "control execution result frame=%s executor=%s sent=%s dx=%.1f dy=%.1f message=%s meta=%s",
                context.frame_id,
                self.last_execution.get("executor_id"),
                self.last_execution.get("sent"),
                float(self.last_execution.get("output_dx") or 0.0),
                float(self.last_execution.get("output_dy") or 0.0),
                self.last_execution.get("message"),
                self.last_execution.get("metadata"),
            )
        elif isinstance(self.last_control, dict):
            if isinstance(self.last_execution, dict):
                self._attach_execution_to_last_control(self.last_execution)
            else:
                self.last_control.setdefault("execution", None)
                self.last_control.setdefault("driver_counts", None)
                self.last_control.setdefault("driver_dx", None)
                self.last_control.setdefault("driver_dy", None)
        self._record_control_frame()
        return RuntimeFrameResult(
            control_intents=control_intents,
            execution_results=execution_results,
        )

    def process_control_tick(self) -> RuntimeFrameResult:
        if str(getattr(self.config.control, "strategy", "pid")) != "experimental_angle_pid":
            return self._empty_runtime_frame_result()
        context = self.last_frame_context
        if context is None:
            return self._empty_runtime_frame_result()
        control_hz = max(1.0, float(getattr(self.config.control, "experimental_angle_control_hz", 60.0)))
        now_ns = time.monotonic_ns()
        min_interval_ns = int(1_000_000_000 / control_hz)
        if self._last_control_tick_ns and now_ns - self._last_control_tick_ns < min_interval_ns:
            return self._empty_runtime_frame_result()
        return self.process_frame(context)

    @staticmethod
    def _empty_runtime_frame_result() -> RuntimeFrameResult:
        return RuntimeFrameResult(control_intents=[], execution_results=[], observation_updated=False)

    def process_detection_batch(
        self,
        detection_batch: DetectionBatch,
        *,
        width: int,
        height: int,
        source_width: int | None = None,
        source_height: int | None = None,
        roi_offset_x: int | None = None,
        roi_offset_y: int | None = None,
    ) -> RuntimeFrameResult:
        total_start_ns = time.monotonic_ns()
        (
            resolved_source_width,
            resolved_source_height,
            resolved_roi_offset_x,
            resolved_roi_offset_y,
            source_geometry_trusted,
            source_geometry_source,
        ) = self._detection_batch_geometry(
            width=int(width),
            height=int(height),
            source_width=source_width,
            source_height=source_height,
            roi_offset_x=roi_offset_x,
            roi_offset_y=roi_offset_y,
        )
        if detection_batch.coordinate_space != "roi":
            self._reset_runtime_control_state("DETECTION_BATCH_COORDINATE_SPACE_INVALID")
            self.last_inference_reason = "DetectionBatch coordinate_space must be roi"
            self._set_detection_batch_pipeline_timings(
                detection_batch,
                total_start_ns=total_start_ns,
                control_start_ns=None,
                done_ns=time.monotonic_ns(),
            )
            self.last_inference_status = self._detection_batch_status_payload(
                detection_batch,
                width=int(width),
                height=int(height),
                now_ns=time.monotonic_ns(),
                reason=self.last_inference_reason,
                available=False,
                mapped_detections=0,
                source_width=resolved_source_width,
                source_height=resolved_source_height,
                source_geometry_source=source_geometry_source,
                source_geometry_trusted=source_geometry_trusted,
                roi_offset_x=resolved_roi_offset_x,
                roi_offset_y=resolved_roi_offset_y,
            )
            return self._empty_runtime_frame_result()
        contract_reason = self._detection_batch_roi_contract_reason(
            detection_batch,
            width=int(width),
            height=int(height),
        )
        if contract_reason:
            self._reset_runtime_control_state("DETECTION_BATCH_ROI_CONTRACT_INVALID")
            self.last_inference_reason = contract_reason
            self._set_detection_batch_pipeline_timings(
                detection_batch,
                total_start_ns=total_start_ns,
                control_start_ns=None,
                done_ns=time.monotonic_ns(),
            )
            self.last_inference_status = self._detection_batch_status_payload(
                detection_batch,
                width=int(width),
                height=int(height),
                now_ns=time.monotonic_ns(),
                reason=self.last_inference_reason,
                available=False,
                mapped_detections=0,
                source_width=resolved_source_width,
                source_height=resolved_source_height,
                source_geometry_source=source_geometry_source,
                source_geometry_trusted=source_geometry_trusted,
                roi_offset_x=resolved_roi_offset_x,
                roi_offset_y=resolved_roi_offset_y,
            )
            return self._empty_runtime_frame_result()
        timestamp_source_reason = self._detection_batch_timestamp_source_reason(detection_batch)
        if timestamp_source_reason:
            self._reset_runtime_control_state("DETECTION_BATCH_TIMESTAMP_SOURCE_INVALID")
            self.last_inference_reason = timestamp_source_reason
            self._set_detection_batch_pipeline_timings(
                detection_batch,
                total_start_ns=total_start_ns,
                control_start_ns=None,
                done_ns=time.monotonic_ns(),
            )
            metadata = getattr(detection_batch, "metadata", {}) or {}
            self.last_inference_status = self._detection_batch_status_payload(
                detection_batch,
                width=int(width),
                height=int(height),
                now_ns=total_start_ns,
                reason=self.last_inference_reason,
                available=False,
                mapped_detections=0,
                source_width=resolved_source_width,
                source_height=resolved_source_height,
                source_geometry_source=source_geometry_source,
                source_geometry_trusted=source_geometry_trusted,
                roi_offset_x=resolved_roi_offset_x,
                roi_offset_y=resolved_roi_offset_y,
                extra={
                    "timestamp_source_invalid": True,
                    "timestamp_source": str(metadata.get("timestamp_source") or ""),
                },
            )
            return self._empty_runtime_frame_result()
        stale_reason = self._detection_batch_stale_reason(
            detection_batch,
            now_ns=total_start_ns,
        )
        if stale_reason:
            self._reset_runtime_control_state("DETECTION_BATCH_STALE")
            self.last_inference_reason = stale_reason
            self._set_detection_batch_pipeline_timings(
                detection_batch,
                total_start_ns=total_start_ns,
                control_start_ns=None,
                done_ns=time.monotonic_ns(),
            )
            self.last_inference_status = self._detection_batch_status_payload(
                detection_batch,
                width=int(width),
                height=int(height),
                now_ns=total_start_ns,
                reason=self.last_inference_reason,
                available=False,
                mapped_detections=0,
                source_width=resolved_source_width,
                source_height=resolved_source_height,
                source_geometry_source=source_geometry_source,
                source_geometry_trusted=source_geometry_trusted,
                roi_offset_x=resolved_roi_offset_x,
                roi_offset_y=resolved_roi_offset_y,
                extra={"stale_rejected": True},
            )
            return self._empty_runtime_frame_result()
        missing_tensor_reason = self._detection_batch_missing_tensor_reason(detection_batch)
        if missing_tensor_reason:
            self._reset_runtime_control_state("DETECTION_BATCH_MISSING_TENSOR_META")
            self.last_inference_reason = missing_tensor_reason
            self._set_detection_batch_pipeline_timings(
                detection_batch,
                total_start_ns=total_start_ns,
                control_start_ns=None,
                done_ns=time.monotonic_ns(),
            )
            self.last_inference_status = self._detection_batch_status_payload(
                detection_batch,
                width=int(width),
                height=int(height),
                now_ns=total_start_ns,
                reason=self.last_inference_reason,
                available=False,
                mapped_detections=0,
                source_width=resolved_source_width,
                source_height=resolved_source_height,
                source_geometry_source=source_geometry_source,
                source_geometry_trusted=source_geometry_trusted,
                roi_offset_x=resolved_roi_offset_x,
                roi_offset_y=resolved_roi_offset_y,
                extra={"missing_tensor_meta": True},
            )
            return self._empty_runtime_frame_result()
        self.last_inference_reason = ""
        self.last_inference_status = self._detection_batch_status_payload(
            detection_batch,
            width=int(width),
            height=int(height),
            now_ns=total_start_ns,
            reason="",
            available=True,
            mapped_detections=len(detection_batch.detections),
            source_width=resolved_source_width,
            source_height=resolved_source_height,
            source_geometry_source=source_geometry_source,
            source_geometry_trusted=source_geometry_trusted,
            roi_offset_x=resolved_roi_offset_x,
            roi_offset_y=resolved_roi_offset_y,
            extra={
                "configured_roi_offset_x": int(getattr(self.config.roi, "offset_x", 0)),
                "configured_roi_offset_y": int(getattr(self.config.roi, "offset_y", 0)),
            },
        )
        context = detection_batch_to_frame_context(
            detection_batch,
            width=int(width),
            height=int(height),
        )
        control_start_ns = time.monotonic_ns()
        result = self.update_control_observation(context)
        done_ns = time.monotonic_ns()
        self._set_detection_batch_pipeline_timings(
            detection_batch,
            total_start_ns=total_start_ns,
            control_start_ns=control_start_ns,
            done_ns=done_ns,
        )
        self._record_control_frame()
        return result

    def _set_detection_batch_pipeline_timings(
        self,
        detection_batch: DetectionBatch,
        *,
        total_start_ns: int,
        control_start_ns: int | None,
        done_ns: int,
    ) -> None:
        handoff_start_ns = int(control_start_ns if control_start_ns is not None else done_ns)
        self.last_pipeline_timings = {
            "roi_ms": 0.0,
            "engine_ms": detection_batch.inference_latency_ms,
            "engine_execute_ms": detection_batch.inference_latency_ms,
            "capture_to_tensor_meta_ms": detection_batch.inference_latency_ms,
            "decode_ms": 0.0,
            "handoff_ms": max(
                0.0,
                (handoff_start_ns - int(detection_batch.inference_end_ts_ns)) / 1e6,
            ),
            "postprocess_ms": 0.0,
            "control_ms": (
                max(0.0, (int(done_ns) - int(control_start_ns)) / 1e6)
                if control_start_ns is not None
                else 0.0
            ),
            "total_ms": max(0.0, (int(done_ns) - int(total_start_ns)) / 1e6),
        }

    def _detection_batch_status_payload(
        self,
        detection_batch: DetectionBatch,
        *,
        width: int,
        height: int,
        now_ns: int,
        reason: str,
        available: bool,
        mapped_detections: int,
        source_width: int,
        source_height: int,
        source_geometry_source: str,
        source_geometry_trusted: bool,
        roi_offset_x: int,
        roi_offset_y: int,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "frame_id": detection_batch.frame_id,
            "capture_ts_ns": detection_batch.capture_ts_ns,
            "frame_age_ms": self._detection_batch_age_ms(detection_batch, now_ns=now_ns),
            "ran": True,
            "available": bool(available),
            "reason": str(reason),
            "detection_coordinate_space": detection_batch.coordinate_space,
            "raw_detections": len(detection_batch.detections),
            "mapped_detections": int(mapped_detections),
            "detection_batch_frame_id": detection_batch.frame_id,
            "detection_batch_capture_ts_ns": detection_batch.capture_ts_ns,
            "inference_start_ts_ns": detection_batch.inference_start_ts_ns,
            "inference_end_ts_ns": detection_batch.inference_end_ts_ns,
            "detection_batch_inference_latency_ms": detection_batch.inference_latency_ms,
            "detection_batch_metadata": dict(getattr(detection_batch, "metadata", {}) or {}),
            "capture_to_tensor_meta_ms": detection_batch.inference_latency_ms,
            "latency_source": "capture_to_tensor_meta_done",
            "classes": list(detection_batch.classes),
            "input_width": int(width),
            "input_height": int(height),
            "model_input_width": int(width),
            "model_input_height": int(height),
            "source": "detection_batch",
            "source_width": int(source_width),
            "source_height": int(source_height),
            "source_geometry_source": str(source_geometry_source),
            "source_geometry_trusted": bool(source_geometry_trusted),
            "roi_offset_x": int(roi_offset_x),
            "roi_offset_y": int(roi_offset_y),
            "roi_region": {
                "x": int(roi_offset_x),
                "y": int(roi_offset_y),
                "w": int(width),
                "h": int(height),
            },
        }
        if extra:
            payload.update(extra)
        return payload

    def _detection_batch_geometry(
        self,
        *,
        width: int,
        height: int,
        source_width: int | None = None,
        source_height: int | None = None,
        roi_offset_x: int | None = None,
        roi_offset_y: int | None = None,
    ) -> tuple[int, int, int, int, bool, str]:
        if source_width is not None and source_height is not None:
            parsed_source_width = int(source_width or 0)
            parsed_source_height = int(source_height or 0)
            if parsed_source_width > 0 and parsed_source_height > 0:
                return (
                    parsed_source_width,
                    parsed_source_height,
                    max(0, int(roi_offset_x or 0)),
                    max(0, int(roi_offset_y or 0)),
                    True,
                    "detection_source",
                )
        capture_config = getattr(self.config, "capture", None)
        config_source_width = int(getattr(capture_config, "width", 0) or 0)
        config_source_height = int(getattr(capture_config, "height", 0) or 0)
        if config_source_width <= 0 or config_source_height <= 0:
            return 0, 0, 0, 0, False, "missing_source_geometry"
        roi_config = getattr(self.config, "roi", None)
        requested_size = max(1, min(int(width), int(height)))
        roi_x, roi_y, _roi_size = center_roi_region(
            source_width=config_source_width,
            source_height=config_source_height,
            requested_size=requested_size,
            offset_x=int(getattr(roi_config, "offset_x", 0) or 0),
            offset_y=int(getattr(roi_config, "offset_y", 0) or 0),
        )
        return config_source_width, config_source_height, int(roi_x), int(roi_y), True, "runtime_config"

    @staticmethod
    def _detection_batch_roi_contract_reason(
        detection_batch: DetectionBatch,
        *,
        width: int,
        height: int,
    ) -> str:
        roi_width = max(1.0, float(width))
        roi_height = max(1.0, float(height))
        for index, detection in enumerate(detection_batch.detections):
            center_x = (float(detection.x1) + float(detection.x2)) * 0.5
            center_y = (float(detection.y1) + float(detection.y2)) * 0.5
            values = (
                float(detection.score),
                float(detection.x1),
                float(detection.y1),
                float(detection.x2),
                float(detection.y2),
                center_x,
                center_y,
            )
            if not all(math.isfinite(value) for value in values):
                return f"DetectionBatch detection contains non-finite value at index {index}"
            if detection.score < 0.0 or detection.score > 1.0:
                return (
                    "DetectionBatch detection score out of range "
                    f"at index {index}: {float(detection.score):.3f}"
                )
            if detection.x2 <= detection.x1 or detection.y2 <= detection.y1:
                return (
                    "DetectionBatch detection box must have positive width and height "
                    f"at index {index}"
                )
            if center_x < 0.0 or center_x > roi_width or center_y < 0.0 or center_y > roi_height:
                return (
                    "DetectionBatch detection center must be inside ROI "
                    f"at index {index}: {center_x:.1f},{center_y:.1f}"
                )
        return ""

    @staticmethod
    def _detection_batch_missing_tensor_reason(detection_batch: DetectionBatch) -> str:
        metadata = getattr(detection_batch, "metadata", {}) or {}
        if str(metadata.get("empty_reason") or "") == "missing_tensor_meta":
            return "DeepStream tensor meta missing for frame"
        return ""

    @staticmethod
    def _detection_batch_timestamp_source_reason(detection_batch: DetectionBatch) -> str:
        metadata = getattr(detection_batch, "metadata", {}) or {}
        if str(metadata.get("source") or "") != "deepstream":
            return ""
        timestamp_source = str(metadata.get("timestamp_source") or "")
        if timestamp_source == "gst_clock_base_time_pts":
            return ""
        return (
            "DeepStream DetectionBatch timestamp_source must be "
            "gst_clock_base_time_pts for control input "
            f"(got {timestamp_source or '-'})"
        )

    def _detection_batch_stale_reason(
        self,
        detection_batch: DetectionBatch,
        *,
        now_ns: int,
    ) -> str:
        threshold_ms = float(
            getattr(getattr(self.config, "control", None), "latency_reject_if_age_exceeds_ms", 0.0)
            or 0.0
        )
        if threshold_ms <= 0.0:
            return ""
        age_ms = self._detection_batch_age_ms(detection_batch, now_ns=now_ns)
        if not math.isfinite(age_ms):
            return ""
        metadata = getattr(detection_batch, "metadata", {}) or {}
        metadata_source = str(metadata.get("source") or "")
        timestamp_source = str(metadata.get("timestamp_source") or "")
        explicit_runtime_clock = metadata_source == "deepstream" or timestamp_source in {
            "gst_clock_base_time_pts",
            "first_probe_offset_pts",
            "observed_probe_time_invalid_pts",
        }
        # Some legacy unit seams use tiny synthetic timestamps. Enforce stale
        # rejection on explicit runtime-clock batches, and otherwise only when
        # capture_ts_ns is plausibly in this process monotonic domain.
        max_plausible_age_ms = max(3_600_000.0, threshold_ms * 100.0)
        if not explicit_runtime_clock and age_ms > max_plausible_age_ms:
            return ""
        if age_ms > threshold_ms:
            return (
                "DetectionBatch frame age exceeds control latency guard: "
                f"{age_ms:.1f}ms > {threshold_ms:.1f}ms"
            )
        return ""

    @staticmethod
    def _detection_batch_age_ms(
        detection_batch: DetectionBatch,
        *,
        now_ns: int,
    ) -> float:
        try:
            capture_ts_ns = int(detection_batch.capture_ts_ns)
        except Exception:
            return 0.0
        return max(0.0, (int(now_ns) - capture_ts_ns) / 1e6)

    def update_control_observation(self, context: FrameContext) -> RuntimeFrameResult:
        if str(getattr(self.config.control, "strategy", "pid")) != "experimental_angle_pid":
            return self.process_frame(context)
        with self._control_lock:
            selection = self._select_control_target(context)
            target = selection.target
            if target is None:
                self._clear_pending_commands("TARGET_UNAVAILABLE")
                prediction_context_available = self.last_frame_context is not None
                if not prediction_context_available:
                    self.last_frame_context = context
                    self.last_target = None
                selector_debug = dict(getattr(self.target_selector, "last_debug", {}) or {})
                track_diagnostics = self._track_diagnostics_payload(
                    selector_debug,
                    selection=selection,
                )
                self.last_control = {
                    "frame_id": context.frame_id,
                    "capture_ts_ns": context.capture_ts_ns,
                    "global_state": self._global_state_from_selection_state(selection.state),
                    "selector_state": selection.state,
                    "selection_reason": selection.reason,
                    "candidates": selection.candidates,
                    "inside_fov": selection.inside_fov,
                    "lost_count": selection.lost_count,
                    "selector_debug": selector_debug,
                    "candidate_filter": self._candidate_filter_payload(selector_debug),
                    "track_diagnostics": track_diagnostics,
                    "will_emit": False,
                    "frame_age_ms": self._frame_age_ms(context),
                    "observation_only": True,
                    "prediction_context_available": prediction_context_available,
                }
                self.last_execution = None
                return RuntimeFrameResult(control_intents=[], execution_results=[], observation_updated=True)
            self.last_frame_context = context
            center = (context.width / 2, context.height / 2)
            target_key = self._control_target_key(target, context)
            strategy_metadata = self._strategy_frame_metadata(context)
            selector_debug = dict(getattr(self.target_selector, "last_debug", {}) or {})
            track_diagnostics = self._track_diagnostics_payload(
                selector_debug,
                target=target,
                selection=selection,
            )
            strategy_metadata.update(
                self._aim_latency_metadata(
                    context=context,
                    target=target,
                    strategy_metadata=strategy_metadata,
                    selector_debug=selector_debug,
                    compute_ts_ns=time.monotonic_ns(),
                )
            )
            strategy_input = self._with_strategy_target(
                BoxInputState(left=True, raw={"mode": "observation_update"}),
                target_key,
                frame_age_ms=self._frame_age_ms(context),
                frame_id=context.frame_id,
                capture_ts_ns=context.capture_ts_ns,
                metadata=strategy_metadata,
            )
            observe = getattr(self.control_strategy, "observe", None)
            observer_debug = observe(target, center, strategy_input) if callable(observe) else {}
            self.last_target = {
                **self._target_payload(target, context),
                "target_detection_index": self._target_detection_index(context, target),
                "target_key": target_key,
                "selector_state": selection.state,
                "selection_reason": selection.reason,
                "locked": selection.locked,
                "priority_rank": selection.priority_rank,
                "distance_px": selection.distance_px,
                "quality_score": selection.quality_score,
                "inside_fov": selection.inside_fov,
                "candidates": selection.candidates,
                "capture_ts_ns": context.capture_ts_ns,
                "frame_age_ms": self._frame_age_ms(context),
                "candidate_filter": self._candidate_filter_payload(selector_debug),
                "track_diagnostics": track_diagnostics,
                "aim_point": strategy_metadata.get("aim_point"),
                "estimated_target_state": strategy_metadata.get("estimated_target_state"),
                "compensated_target": strategy_metadata.get("compensated_target"),
            }
            self.last_control = {
                "frame_id": context.frame_id,
                "capture_ts_ns": context.capture_ts_ns,
                "global_state": self._global_state_from_selection_state(selection.state),
                "selector_state": selection.state,
                "selection_reason": selection.reason,
                "selector_debug": dict(getattr(self.target_selector, "last_debug", {}) or {}),
                "candidate_filter": self._candidate_filter_payload(selector_debug),
                "track_diagnostics": track_diagnostics,
                "target_key": target_key,
                "target_detection_index": self._target_detection_index(context, target),
                "priority_rank": selection.priority_rank,
                "distance_px": selection.distance_px,
                "quality_score": selection.quality_score,
                "inside_fov": selection.inside_fov,
                "candidates": selection.candidates,
                "pipeline": {
                    "tracker": observer_debug,
                    "aim_point": strategy_metadata.get("aim_point"),
                    "estimated_target_state": strategy_metadata.get("estimated_target_state"),
                    "compensated_target": strategy_metadata.get("compensated_target"),
                    "latency_compensation": strategy_metadata.get("latency_compensation"),
                },
                "aim_point": strategy_metadata.get("aim_point"),
                "estimated_target_state": strategy_metadata.get("estimated_target_state"),
                "compensated_target": strategy_metadata.get("compensated_target"),
                "latency_compensation": strategy_metadata.get("latency_compensation"),
                "will_emit": False,
                "observation_only": True,
            }
            self.last_execution = None
        return RuntimeFrameResult(control_intents=[], execution_results=[], observation_updated=True)

    def process_captured_frame(self, frame: CapturedFrame) -> RuntimeFrameResult:
        total_start_ns = time.monotonic_ns()
        if self.inference is None:
            self._record_inference_status(
                frame=frame,
                ran=False,
                available=False,
                reason="推理运行时未初始化",
            )
            result = self.update_control_observation(self._empty_frame_context(frame))
            self._record_pipeline_timings(total_start_ns, control_start_ns=total_start_ns)
            return result

        infer = getattr(self.inference, "infer", None)
        if not callable(infer):
            self._record_inference_status(
                frame=frame,
                ran=False,
                available=False,
                reason="推理运行时没有 infer 方法",
            )
            result = self.update_control_observation(self._empty_frame_context(frame))
            self._record_pipeline_timings(total_start_ns, control_start_ns=total_start_ns)
            return result

        try:
            roi_start_ns = time.monotonic_ns()
            roi_frame = center_roi_frame(
                frame,
                requested_size=self.config.roi.size,
                offset_x=self.config.roi.offset_x,
                offset_y=self.config.roi.offset_y,
            )
            infer_start_ns = time.monotonic_ns()
            inference_result = infer(roi_frame)
            postprocess_start_ns = time.monotonic_ns()
        except Exception as exc:
            self.last_inference_reason = str(exc)
            self._record_inference_status(
                frame=frame,
                roi_frame=locals().get("roi_frame"),
                ran=True,
                available=False,
                reason=str(exc),
            )
            control_start_ns = time.monotonic_ns()
            result = self.update_control_observation(self._empty_frame_context(frame))
            self._record_pipeline_timings(
                total_start_ns,
                roi_start_ns=locals().get("roi_start_ns"),
                infer_start_ns=locals().get("infer_start_ns"),
                postprocess_start_ns=locals().get("postprocess_start_ns"),
                control_start_ns=control_start_ns,
            )
            return result

        if not isinstance(inference_result, InferenceResult):
            self.last_inference_reason = "invalid inference result"
            self._record_inference_status(
                frame=frame,
                roi_frame=roi_frame,
                ran=True,
                available=False,
                reason=self.last_inference_reason,
            )
            control_start_ns = time.monotonic_ns()
            result = self.update_control_observation(self._empty_frame_context(frame))
            self._record_pipeline_timings(
                total_start_ns,
                roi_start_ns=roi_start_ns,
                infer_start_ns=infer_start_ns,
                postprocess_start_ns=postprocess_start_ns,
                control_start_ns=control_start_ns,
            )
            return result

        if not inference_result.available:
            self.last_inference_reason = inference_result.reason
            self._record_inference_status(
                frame=frame,
                roi_frame=roi_frame,
                ran=True,
                available=False,
                reason=inference_result.reason,
                raw_detections=len(inference_result.detections),
                debug=inference_result.debug,
            )
            control_start_ns = time.monotonic_ns()
            result = self.update_control_observation(self._empty_frame_context(frame))
            self._record_pipeline_timings(
                total_start_ns,
                roi_start_ns=roi_start_ns,
                infer_start_ns=infer_start_ns,
                postprocess_start_ns=postprocess_start_ns,
                control_start_ns=control_start_ns,
            )
            return result

        try:
            detections: list[Detection] = []
            for item in inference_result.detections:
                detections.append(
                    Detection(
                        cls=item.cls,
                        score=item.score,
                        box=item.box,
                    )
                )
            classes = list(inference_result.classes)
            detection_batch = DetectionBatch(
                frame_id=frame.frame_id,
                capture_ts_ns=frame.capture_ts_ns,
                inference_start_ts_ns=infer_start_ns,
                inference_end_ts_ns=postprocess_start_ns,
                detections=detections,
                classes=classes,
                coordinate_space="roi",
            )
        except Exception as exc:
            self.last_inference_reason = str(exc)
            self._record_inference_status(
                frame=frame,
                roi_frame=roi_frame,
                ran=True,
                available=False,
                reason=str(exc),
                raw_detections=len(inference_result.detections),
                debug=inference_result.debug,
            )
            control_start_ns = time.monotonic_ns()
            result = self.update_control_observation(self._empty_frame_context(frame))
            self._record_pipeline_timings(
                total_start_ns,
                roi_start_ns=roi_start_ns,
                infer_start_ns=infer_start_ns,
                postprocess_start_ns=postprocess_start_ns,
                control_start_ns=control_start_ns,
            )
            return result
        self.last_inference_reason = ""
        self._record_inference_status(
            frame=frame,
            roi_frame=roi_frame,
            ran=True,
            available=True,
            reason="",
            raw_detections=len(inference_result.detections),
            mapped_detections=len(detection_batch.detections),
            classes=detection_batch.classes,
            detection_batch=detection_batch,
            debug=inference_result.debug,
        )

        context = FrameContext(
            frame_id=detection_batch.frame_id,
            width=roi_frame.width,
            height=roi_frame.height,
            detections=detection_batch.detections,
            classes=detection_batch.classes,
            capture_ts_ns=detection_batch.capture_ts_ns,
        )
        control_start_ns = time.monotonic_ns()
        result = self.update_control_observation(context)
        self._record_pipeline_timings(
            total_start_ns,
            roi_start_ns=roi_start_ns,
            infer_start_ns=infer_start_ns,
            postprocess_start_ns=postprocess_start_ns,
            control_start_ns=control_start_ns,
        )
        return result

    def _record_pipeline_timings(
        self,
        total_start_ns: int,
        *,
        roi_start_ns: int | None = None,
        infer_start_ns: int | None = None,
        postprocess_start_ns: int | None = None,
        control_start_ns: int,
    ) -> None:
        done_ns = time.monotonic_ns()

        def ms(start: int | None, end: int | None) -> float:
            if start is None or end is None:
                return 0.0
            return max(0.0, (end - start) / 1e6)

        inference_debug = self.last_inference_status.get("debug", {})
        debug_timings = inference_debug.get("timings", {}) if isinstance(inference_debug, dict) else {}
        decode_debug = inference_debug.get("decode", {}) if isinstance(inference_debug, dict) else {}
        decode_timings = decode_debug.get("timings", {}) if isinstance(decode_debug, dict) else {}
        decode_ms = float(decode_timings.get("decode_ms") or debug_timings.get("decode_ms") or 0.0)
        engine_total_ms = float(decode_timings.get("total_ms") or debug_timings.get("execute_total_ms") or 0.0)
        engine_execute_ms = max(0.0, engine_total_ms - decode_ms)
        self.last_pipeline_timings = {
            "roi_ms": ms(roi_start_ns, infer_start_ns),
            "engine_ms": ms(infer_start_ns, postprocess_start_ns),
            "engine_execute_ms": engine_execute_ms,
            "decode_ms": decode_ms,
            "postprocess_ms": ms(postprocess_start_ns, control_start_ns),
            "control_ms": ms(control_start_ns, done_ns),
            "total_ms": ms(total_start_ns, done_ns),
        }
        self._record_control_frame()

    def _empty_frame_context(self, frame: CapturedFrame) -> FrameContext:
        return FrameContext(
            frame_id=frame.frame_id,
            width=self._source_width(frame),
            height=self._source_height(frame),
            capture_ts_ns=frame.capture_ts_ns,
        )

    def _record_inference_status(
        self,
        *,
        frame: CapturedFrame,
        ran: bool,
        available: bool,
        reason: str,
        roi_frame: Any | None = None,
        raw_detections: int = 0,
        mapped_detections: int = 0,
        classes: list[str] | None = None,
        detection_batch: DetectionBatch | None = None,
        debug: dict[str, Any] | None = None,
    ) -> None:
        debug_payload = dict(debug or {})
        preprocess = debug_payload.get("preprocess")
        preprocess_debug = preprocess if isinstance(preprocess, dict) else {}
        model_input_width = self._debug_int(preprocess_debug, "model_width")
        model_input_height = self._debug_int(preprocess_debug, "model_height")
        model_to_roi_scale_x = self._debug_float(preprocess_debug, "model_to_roi_scale_x")
        model_to_roi_scale_y = self._debug_float(preprocess_debug, "model_to_roi_scale_y")
        input_downscale_factor = self._debug_float(preprocess_debug, "downscale_factor")
        input_pixel_ratio = self._debug_float(preprocess_debug, "pixel_ratio")
        (
            source_width,
            source_height,
            source_geometry_source,
            source_geometry_trusted,
        ) = self._capture_geometry(frame)
        frame_resource = getattr(frame, "frame_resource", None)
        input_resource = (
            getattr(roi_frame, "frame_resource", None) if roi_frame is not None else None
        )
        if input_resource is None:
            input_resource = frame_resource
        self.last_inference_status = {
            "frame_id": frame.frame_id,
            "capture_ts_ns": frame.capture_ts_ns,
            "frame_receive_ts_ns": int(getattr(frame, "receive_ts_ns", frame.capture_ts_ns)),
            "frame_source_ts_ns": getattr(frame, "source_ts_ns", None),
            "frame_source_ts_kind": str(getattr(frame, "source_ts_kind", "") or ""),
            "frame_userspace_process_ms": float(getattr(frame, "userspace_process_ms", 0.0) or 0.0),
            "frame_resource_kind": str(getattr(frame_resource, "kind", "") or ""),
            "frame_resource_memory": str(getattr(frame_resource, "memory", "") or "cpu"),
            "frame_dmabuf_fd": getattr(frame_resource, "dmabuf_fd", None),
            "frame_age_ms": max(0.0, (time.monotonic_ns() - int(frame.capture_ts_ns)) / 1e6),
            "ran": ran,
            "available": available,
            "reason": reason,
            "raw_detections": raw_detections,
            "mapped_detections": mapped_detections,
            "detection_batch_frame_id": detection_batch.frame_id if detection_batch is not None else 0,
            "detection_batch_capture_ts_ns": detection_batch.capture_ts_ns if detection_batch is not None else 0,
            "inference_start_ts_ns": detection_batch.inference_start_ts_ns if detection_batch is not None else 0,
            "inference_end_ts_ns": detection_batch.inference_end_ts_ns if detection_batch is not None else 0,
            "detection_batch_inference_latency_ms": detection_batch.inference_latency_ms if detection_batch is not None else 0.0,
            "classes": list(classes or []),
            "input_width": int(getattr(roi_frame, "width", frame.width)),
            "input_height": int(getattr(roi_frame, "height", frame.height)),
            "input_image_width": self._image_width(getattr(roi_frame, "image", frame.image)),
            "input_image_height": self._image_height(getattr(roi_frame, "image", frame.image)),
            "input_pixel_format": str(getattr(roi_frame, "pixel_format", frame.pixel_format)),
            "input_resource_kind": str(getattr(input_resource, "kind", "") or ""),
            "input_resource_memory": str(getattr(input_resource, "memory", "") or "cpu"),
            "input_dmabuf_fd": getattr(input_resource, "dmabuf_fd", None),
            "model_input_width": model_input_width,
            "model_input_height": model_input_height,
            "model_to_roi_scale_x": model_to_roi_scale_x,
            "model_to_roi_scale_y": model_to_roi_scale_y,
            "input_downscale_factor": input_downscale_factor,
            "input_pixel_ratio": input_pixel_ratio,
            "input_density_warning": bool(preprocess_debug.get("density_warning", False)),
            "detection_coordinate_space": detection_batch.coordinate_space if detection_batch is not None else "roi",
            "source_width": source_width,
            "source_height": source_height,
            "source_geometry_source": source_geometry_source,
            "source_geometry_trusted": source_geometry_trusted,
            "roi_offset_x": int(getattr(roi_frame, "offset_x", 0)),
            "roi_offset_y": int(getattr(roi_frame, "offset_y", 0)),
            "roi_region": {
                "x": int(getattr(roi_frame, "offset_x", 0)),
                "y": int(getattr(roi_frame, "offset_y", 0)),
                "w": int(getattr(roi_frame, "width", frame.width)),
                "h": int(getattr(roi_frame, "height", frame.height)),
            },
            "configured_roi_offset_x": int(getattr(self.config.roi, "offset_x", 0)),
            "configured_roi_offset_y": int(getattr(self.config.roi, "offset_y", 0)),
            "debug": debug_payload,
        }

    @staticmethod
    def _debug_int(payload: dict[str, Any], key: str) -> int:
        value = payload.get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    @staticmethod
    def _debug_float(payload: dict[str, Any], key: str) -> float:
        value = payload.get(key)
        return float(value) if isinstance(value, (int, float)) else 0.0

    def _control_intent_from_context(self, context: FrameContext) -> ControlIntent | None:
        frame_age_ms = self._frame_age_ms(context)
        selection = self._select_control_target(context)
        target = selection.target
        if target is None:
            self._clear_pending_commands("TARGET_UNAVAILABLE")
            self._log_no_control_target(context, selection)
            self.last_target = None
            selector_debug = dict(getattr(self.target_selector, "last_debug", {}) or {})
            track_diagnostics = self._track_diagnostics_payload(
                selector_debug,
                selection=selection,
            )
            self.last_control = {
                "frame_id": context.frame_id,
                "global_state": self._global_state_from_selection_state(selection.state),
                "selector_state": selection.state,
                "selection_reason": selection.reason,
                "candidates": selection.candidates,
                "inside_fov": selection.inside_fov,
                "lost_count": selection.lost_count,
                "selector_debug": selector_debug,
                "candidate_filter": self._candidate_filter_payload(selector_debug),
                "track_diagnostics": track_diagnostics,
                "will_emit": False,
                "frame_age_ms": frame_age_ms,
                "capture_ts_ns": context.capture_ts_ns,
            }
            self.last_execution = None
            return None

        center = (context.width / 2, context.height / 2)
        trigger_mode = str(getattr(self.config.control, "trigger_mode", "hardware") or "hardware")
        if trigger_mode not in {"hardware", "always"}:
            trigger_mode = "hardware"
        box_input = (
            BoxInputState(left=True, raw={"source": "always", "active": True, "reason": "trigger mode always"})
            if trigger_mode == "always"
            else self._box_input_state()
        )
        target_key = self._control_target_key(target, context)
        calibration_status = self._calibration_fingerprint_status()
        if calibration_status["control_allowed"] is not True:
            self._clear_pending_commands(str(calibration_status["reason_code"]))
            selector_debug = dict(getattr(self.target_selector, "last_debug", {}) or {})
            track_diagnostics = self._track_diagnostics_payload(
                selector_debug,
                target=target,
                selection=selection,
            )
            target_detection_index = self._target_detection_index(context, target)
            self.last_target = {
                **self._target_payload(target, context),
                "target_detection_index": target_detection_index,
                "target_key": target_key,
                "selector_state": selection.state,
                "selection_reason": selection.reason,
                "locked": selection.locked,
                "priority_rank": selection.priority_rank,
                "distance_px": selection.distance_px,
                "quality_score": selection.quality_score,
                "inside_fov": selection.inside_fov,
                "candidates": selection.candidates,
                "capture_ts_ns": context.capture_ts_ns,
                "frame_age_ms": frame_age_ms,
                "candidate_filter": self._candidate_filter_payload(selector_debug),
                "track_diagnostics": track_diagnostics,
            }
            self.last_control = {
                "frame_id": context.frame_id,
                "capture_ts_ns": context.capture_ts_ns,
                "frame_age_ms": frame_age_ms,
                "global_state": "DISABLED",
                "selector_state": selection.state,
                "selection_reason": selection.reason,
                "selector_debug": selector_debug,
                "candidate_filter": self._candidate_filter_payload(selector_debug),
                "track_diagnostics": track_diagnostics,
                "target_key": target_key,
                "target_detection_index": target_detection_index,
                "priority_rank": selection.priority_rank,
                "distance_px": selection.distance_px,
                "quality_score": selection.quality_score,
                "inside_fov": selection.inside_fov,
                "candidates": selection.candidates,
                "dx": 0.0,
                "dy": 0.0,
                "reason": calibration_status["reason_code"],
                "trigger_reason": calibration_status["reason"],
                "will_emit": False,
                "control_allowed": False,
                "calibration_status": calibration_status,
            }
            self.last_execution = {
                "executor_id": str(getattr(self.executors, "selected", "")),
                "sent": False,
                "accepted": False,
                "clipped": False,
                "output_dx": 0.0,
                "output_dy": 0.0,
                "message": calibration_status["reason"],
                "intent": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "accepted": False,
                    "clipped": False,
                    "reason": calibration_status["reason_code"],
                },
            }
            return None
        strategy_metadata = self._strategy_frame_metadata(context)
        selector_debug = dict(getattr(self.target_selector, "last_debug", {}) or {})
        track_diagnostics = self._track_diagnostics_payload(
            selector_debug,
            target=target,
            selection=selection,
        )
        strategy_metadata.update(
            self._aim_latency_metadata(
                context=context,
                target=target,
                strategy_metadata=strategy_metadata,
                selector_debug=selector_debug,
                compute_ts_ns=time.monotonic_ns(),
            )
        )
        strategy_input = (
            self._with_strategy_target(
                box_input,
                target_key,
                frame_age_ms=frame_age_ms,
                frame_id=context.frame_id,
                capture_ts_ns=context.capture_ts_ns,
                metadata=strategy_metadata,
            )
            if box_input.active
            else BoxInputState(
                left=True,
                raw={
                    "mode": "control_preview_calculation",
                    "target_key": target_key,
                    "frame_age_ms": frame_age_ms,
                    "frame_id": context.frame_id,
                    "capture_ts_ns": context.capture_ts_ns,
                    **strategy_metadata,
                },
            )
        )
        aim_ratio = self._active_aim_ratio()
        aim_x, aim_y = aim_point(target, aim_ratio)
        command = self.control_strategy.calculate(target, center, strategy_input)
        pipeline_debug = dict(command.debug)
        tracker_debug = pipeline_debug.get("tracker")
        predicted_source = (
            isinstance(tracker_debug, dict)
            and tracker_debug.get("new_observation") is False
        )
        strategy_aim_x = pipeline_debug.get("aim_x")
        strategy_aim_y = pipeline_debug.get("aim_y")
        if isinstance(strategy_aim_x, (int, float)) and isinstance(strategy_aim_y, (int, float)):
            aim_x = float(strategy_aim_x)
            aim_y = float(strategy_aim_y)
        intent = ControlIntent(
            dx=command.dx,
            dy=command.dy,
            action="move",
            confidence=command.confidence,
            reason=command.reason,
            source_id="runtime.control",
            move_kind=command.move_kind,
            move_ms=command.move_ms,
            trace_ms=command.trace_ms,
            bezier_ctrl=command.bezier_ctrl,
            source_frame_id=context.frame_id,
            source_track_id=int(getattr(target, "track_id")) if hasattr(target, "track_id") else None,
            predicted_source=predicted_source,
        )
        output_mode = str(
            getattr(self.config.control, "output_mode", "")
            or getattr(getattr(self.config, "executor", None), "default", "")
        )
        hardware_kind = str(getattr(getattr(self.config, "hardware", None), "kind", "none"))
        requires_trigger = trigger_mode != "always"
        control_allowed = bool(pipeline_debug.get("control_allowed", True))
        calibration_status = self._calibration_fingerprint_status()
        if calibration_status["control_allowed"] is not True:
            control_allowed = False
        can_emit = (box_input.active or not requires_trigger) and control_allowed
        trigger_raw = getattr(box_input, "raw", {}) or {}
        trigger_requirement = self._trigger_requirement_label(
            requires_trigger=requires_trigger,
            trigger_mode=trigger_mode,
        )
        aim_error_x = float(aim_x - center[0])
        aim_error_y = float(center[1] - aim_y)
        raw_error_x = float(pipeline_debug.get("raw_px_x", aim_error_x))
        raw_error_y = float(pipeline_debug.get("raw_px_y", aim_error_y))
        raw_error_y_image = -raw_error_y
        target_detection_index = self._target_detection_index(context, target)
        self.last_target = {
            **self._target_payload(target, context),
            "target_detection_index": target_detection_index,
            "aim_ratio": aim_ratio,
            "aim_y_ratio": aim_ratio,
            "aim_x": aim_x,
            "aim_y": aim_y,
            "aim_offset_x": float(aim_x - center[0]),
            "aim_offset_y": float(aim_y - center[1]),
            "selector_state": selection.state,
            "selection_reason": selection.reason,
            "locked": selection.locked,
            "priority_rank": selection.priority_rank,
            "distance_px": selection.distance_px,
            "quality_score": selection.quality_score,
            "inside_fov": selection.inside_fov,
            "candidates": selection.candidates,
            "bbox_age_ms": 0.0,
            "is_stale": False,
            "target_key": target_key,
            "frame_age_ms": frame_age_ms,
            "capture_ts_ns": context.capture_ts_ns,
            "candidate_filter": self._candidate_filter_payload(selector_debug),
            "track_diagnostics": track_diagnostics,
            "aim_point": pipeline_debug.get("aim_point") or strategy_metadata.get("aim_point"),
            "estimated_target_state": pipeline_debug.get("estimated_target_state") or strategy_metadata.get("estimated_target_state"),
            "compensated_target": pipeline_debug.get("compensated_target") or strategy_metadata.get("compensated_target"),
        }
        self.last_control = {
            "frame_id": context.frame_id,
            "capture_ts_ns": context.capture_ts_ns,
            "frame_age_ms": frame_age_ms,
            "global_state": self._global_state_from_selection_state(selection.state),
            "aim_error_x": aim_error_x,
            "aim_error_y": aim_error_y,
            "raw_error_x": raw_error_x,
            "raw_error_y": raw_error_y,
            "raw_error_y_image": raw_error_y_image,
            "aim_ratio": aim_ratio,
            "aim_y_ratio": aim_ratio,
            "aim_x": aim_x,
            "aim_y": aim_y,
            "dx": command.dx,
            "dy": command.dy,
            "confidence": command.confidence,
            "reason": command.reason,
            "pipeline": pipeline_debug,
            "aim_point": pipeline_debug.get("aim_point") or strategy_metadata.get("aim_point"),
            "estimated_target_state": pipeline_debug.get("estimated_target_state") or strategy_metadata.get("estimated_target_state"),
            "compensated_target": pipeline_debug.get("compensated_target") or strategy_metadata.get("compensated_target"),
            "latency_compensation": pipeline_debug.get("latency_compensation") or strategy_metadata.get("latency_compensation"),
            "selector_state": selection.state,
            "selection_reason": selection.reason,
            "selector_debug": dict(getattr(self.target_selector, "last_debug", {}) or {}),
            "candidate_filter": self._candidate_filter_payload(selector_debug),
            "track_diagnostics": track_diagnostics,
            "target_detection_index": target_detection_index,
            "priority_rank": selection.priority_rank,
            "distance_px": selection.distance_px,
            "quality_score": selection.quality_score,
            "trigger_active": box_input.active,
            "trigger_required": requires_trigger,
            "trigger_mode": trigger_mode,
            "trigger_requirement": trigger_requirement,
            "trigger_reason": str(trigger_raw.get("reason") or trigger_raw.get("mode") or trigger_raw.get("source") or ""),
            "trigger_raw": trigger_raw,
            "output_mode": output_mode,
            "will_emit": can_emit,
            "control_allowed": control_allowed,
            "calibration_status": calibration_status,
            "bbox_age_ms": 0.0,
            "is_stale": False,
                "target_key": target_key,
            }
        self._log_control_decision(
            context=context,
            target=target,
            command=command,
            can_emit=can_emit,
            trigger_raw=trigger_raw,
            output_mode=output_mode,
            hardware_kind=hardware_kind,
            trigger_mode=trigger_mode,
        )
        if not can_emit:
            clear_reason = "CONTROL_NOT_ALLOWED" if not control_allowed else "TRIGGER_INACTIVE"
            self._clear_pending_commands(clear_reason)
            self.last_execution = {
                "executor_id": str(getattr(self.executors, "selected", "")),
                "sent": False,
                "accepted": False,
                "clipped": False,
                "output_dx": 0.0,
                "output_dy": 0.0,
                "message": f"{trigger_requirement}，控制量未发送",
                "intent": {
                    "dx": float(command.dx),
                    "dy": float(command.dy),
                    "accepted": False,
                    "clipped": False,
                    "reason": command.reason,
                },
            }
        return intent if can_emit else None

    def _clear_pending_commands(self, reason: str) -> None:
        scheduler = getattr(self.executors, "scheduler", None)
        clear = getattr(scheduler, "clear", None)
        if callable(clear):
            clear(reason)

    @staticmethod
    def _global_state_from_selection_state(state: Any) -> str:
        normalized = str(state or "").strip().lower()
        if normalized in {"fresh", "locked", "switch_committed", "committed_initial"}:
            return "TRACKING"
        if normalized in {"acquire", "acquiring"}:
            return "ACQUIRING"
        if normalized == "predicting":
            return "PREDICTING"
        if normalized in {"identity_uncertain", "switch_pending"}:
            return "IDENTITY_UNCERTAIN"
        if normalized in {"target_unavailable", "missing", "no_target", "reacquire", "lost"}:
            return "TARGET_UNAVAILABLE"
        if normalized == "cooldown":
            return "COOLDOWN"
        if normalized in {"disabled", "reset"}:
            return "DISABLED"
        return "IDLE" if not normalized else normalized.upper()

    @staticmethod
    def _global_state_from_execution_metadata(metadata: dict[str, Any]) -> str | None:
        scheduler = metadata.get("scheduler")
        scheduler_payload = dict(scheduler) if isinstance(scheduler, dict) else {}
        scheduler_execution = scheduler_payload.get("execution")
        scheduler_execution_payload = (
            dict(scheduler_execution) if isinstance(scheduler_execution, dict) else {}
        )
        command_status = str(
            scheduler_execution_payload.get("command_status")
            or scheduler_payload.get("command_status")
            or metadata.get("command_status")
            or ""
        ).strip().lower()
        cancel_reason = str(
            scheduler_execution_payload.get("cancel_reason")
            or scheduler_payload.get("cancel_reason")
            or metadata.get("cancel_reason")
            or ""
        ).strip().upper()
        cooldown = (
            scheduler_execution_payload.get("cooldown") is True
            or scheduler_payload.get("cooldown") is True
            or command_status == "cooldown"
        )
        if cooldown or command_status == "error" or cancel_reason == "DEVICE_ERROR":
            return "COOLDOWN"
        if command_status in {"expired", "rejected"} or cancel_reason in {
            "EXPIRED",
            "TARGET_UNAVAILABLE",
            "CONTROL_NOT_ALLOWED",
        }:
            return "TARGET_UNAVAILABLE"
        return None

    @staticmethod
    def _trigger_requirement_label(*, requires_trigger: bool, trigger_mode: str) -> str:
        if not requires_trigger:
            return "无需触发"
        if trigger_mode == "hardware":
            return "需要 kmNet 硬件按键回传"
        return "需要触发"

    def _log_control_decision(
        self,
        *,
        context: FrameContext,
        target: Track | Detection,
        command: Any,
        can_emit: bool,
        trigger_raw: dict[str, Any],
        output_mode: str,
        hardware_kind: str,
        trigger_mode: str,
    ) -> None:
        trigger_source = str(trigger_raw.get("source") or trigger_raw.get("mode") or trigger_raw.get("reason") or "")
        signature = (
            f"emit={can_emit}|out={output_mode}|hardware={hardware_kind}|"
            f"trigger={trigger_mode}|source={trigger_source}|"
            f"active={bool(trigger_raw.get('left') or trigger_raw.get('right') or trigger_raw.get('active'))}"
        )
        now = time.monotonic()
        if signature == self._last_control_log_signature and now - self._last_control_log_s < 1.0:
            return
        self._last_control_log_signature = signature
        self._last_control_log_s = now
        logger.info(
            "control decision frame=%s age_ms=%.1f target_cls=%s score=%.3f dx=%.1f dy=%.1f emit=%s output=%s hardware=%s trigger=%s trigger_source=%s trigger_raw=%s reason=%s",
            context.frame_id,
            self._frame_age_ms(context),
            int(getattr(target, "cls", -1)),
            float(getattr(target, "score", 0.0)),
            float(getattr(command, "dx", 0.0)),
            float(getattr(command, "dy", 0.0)),
            can_emit,
            output_mode,
            hardware_kind,
            trigger_mode,
            trigger_source,
            trigger_raw,
            str(getattr(command, "reason", "")),
        )

    def _log_no_control_target(self, context: FrameContext, selection: TargetSelection) -> None:
        return

    def _log_box_input_state(self, state: BoxInputState, source: str) -> None:
        raw = getattr(state, "raw", {}) or {}
        signature = (
            f"box|source={source}|active={state.active}|left={state.left}|"
            f"right={state.right}|side={state.side}|raw={raw}"
        )
        if not self._should_log("_last_box_input_log_signature", "_last_box_input_log_s", signature, interval_s=1.0):
            return
        logger.info(
            "box input source=%s active=%s left=%s right=%s side=%s raw=%s",
            source,
            state.active,
            state.left,
            state.right,
            state.side,
            raw,
        )

    def _should_log(self, signature_attr: str, time_attr: str, signature: str, *, interval_s: float) -> bool:
        now = time.monotonic()
        previous_signature = str(getattr(self, signature_attr, ""))
        previous_s = float(getattr(self, time_attr, 0.0))
        if signature == previous_signature and now - previous_s < interval_s:
            return False
        setattr(self, signature_attr, signature)
        setattr(self, time_attr, now)
        return True

    def _select_control_target(self, context: FrameContext) -> TargetSelection:
        return self.target_selector.select(
            context,
            min_confidence=float(getattr(self.config.control, "min_confidence", 0.0)),
            fov_ratio=float(getattr(self.config.control, "fov_ratio", 0.28)),
            aim_ratio=self._active_aim_ratio(),
            class_filter=str(getattr(self.config.inference, "detection_class_filter", "all")),
            class_priority=self._class_priority(),
            sticky_bias=float(getattr(self.config.control, "target_sticky_bias", 0.25)),
            lock_enabled=bool(getattr(self.config.control, "target_lock_enabled", True)),
            lost_grace_frames=int(getattr(self.config.control, "target_lost_grace_frames", 5)),
            ratio_max_aspect=float(getattr(self.config.control, "candidate_ratio_max_aspect", 6.0)),
            quality_confidence_weight=float(getattr(self.config.control, "candidate_quality_confidence_weight", 0.7)),
            quality_area_weight=float(getattr(self.config.control, "candidate_quality_area_weight", 0.3)),
            class_priority_quality_margin=float(getattr(self.config.control, "class_priority_quality_margin", 0.08)),
            tracker_confirm_frames=int(getattr(self.config.control, "tracker_confirm_frames", 2)),
            target_switch_min_preference_advantage=float(getattr(self.config.control, "target_switch_min_preference_advantage", 0.08)),
            target_switch_min_continuity_score=float(getattr(self.config.control, "target_switch_min_continuity_score", 0.70)),
            target_switch_confirm_frames=int(getattr(self.config.control, "target_switch_confirm_frames", 3)),
            tracker_matching_distance_px=float(getattr(self.config.control, "tracker_matching_distance_px", 140.0)),
            tracker_ambiguity_margin=float(getattr(self.config.control, "tracker_ambiguity_margin", 0.08)),
            tracker_missing_timeout_ms=float(getattr(self.config.control, "tracker_missing_timeout_ms", 120.0)),
            tracker_delete_timeout_ms=float(getattr(self.config.control, "tracker_delete_timeout_ms", 250.0)),
            tracker_match_threshold=float(getattr(self.config.control, "tracker_match_threshold", 0.65)),
            tracker_mahalanobis_gate=float(getattr(self.config.control, "tracker_mahalanobis_gate", 9.21)),
            kalman_enabled=bool(getattr(self.config.control, "kalman_enabled", True)),
            kalman_acceleration_noise=float(getattr(self.config.control, "kalman_acceleration_noise", 1200.0)),
            kalman_measurement_noise_x=float(getattr(self.config.control, "kalman_measurement_noise_x", 16.0)),
            kalman_measurement_noise_y=float(getattr(self.config.control, "kalman_measurement_noise_y", 16.0)),
            kalman_max_predict_missing_ms=float(getattr(self.config.control, "kalman_max_predict_missing_ms", 80.0)),
            kalman_max_predict_steps=int(getattr(self.config.control, "kalman_max_predict_steps", 5)),
            kalman_max_predict_dt_ms=float(getattr(self.config.control, "kalman_max_predict_dt_ms", 35.0)),
            kalman_max_position_sigma_px=float(getattr(self.config.control, "kalman_max_position_sigma_px", 45.0)),
            kalman_max_covariance_trace=float(getattr(self.config.control, "kalman_max_covariance_trace", 5000.0)),
            kalman_nis_threshold=float(getattr(self.config.control, "kalman_nis_threshold", 9.21)),
            kalman_nis_hard_reject=float(getattr(self.config.control, "kalman_nis_hard_reject", 16.0)),
            kalman_min_identity_confidence=float(getattr(self.config.control, "kalman_min_identity_confidence", 0.70)),
            kalman_min_prediction_confidence=float(getattr(self.config.control, "kalman_min_prediction_confidence", 0.35)),
            kalman_prediction_decay_tau_ms=float(getattr(self.config.control, "kalman_prediction_decay_tau_ms", 45.0)),
        )

    def _active_aim_ratio(self) -> float:
        value = getattr(self.config.control, "aim_ratio", 40.0)
        return max(0.0, min(100.0, float(value)))

    def _filter_detections_by_config(self, detections: list[Detection]) -> list[Detection]:
        selected = str(getattr(self.config.inference, "detection_class_filter", "all"))
        if selected == "all":
            return detections
        try:
            class_id = int(selected)
        except ValueError:
            return detections
        return [item for item in detections if int(item.cls) == class_id]

    def _class_priority(self) -> list[int]:
        raw = str(getattr(self.config.inference, "detection_class_priority", ""))
        result: list[int] = []
        seen: set[int] = set()
        for part in raw.split(","):
            try:
                class_id = int(part.strip())
            except ValueError:
                continue
            if class_id not in seen:
                seen.add(class_id)
                result.append(class_id)
        return result

    def _box_input_state(self) -> BoxInputState:
        if str(getattr(getattr(self.config, "hardware", None), "kind", "none")).lower() != "kmnet":
            state = BoxInputState(raw={"source": "kmnet_executor", "reason": "kmNet hardware is required"})
            self._log_box_input_state(state, "kmnet_required")
            return state
        button_reader = getattr(self.executors, "read_buttons", None)
        buttons: dict[str, Any] | None = None
        if callable(button_reader):
            try:
                buttons = button_reader()
            except Exception as exc:
                buttons = {"available": False, "left": False, "right": False, "reason": f"button reader exception: {exc}"}
                logger.warning("box input button reader failed: %s", exc)
            if buttons.get("available") is True:
                hardware_state = BoxInputState(
                    left=bool(buttons.get("left")),
                    right=bool(buttons.get("right")),
                    raw={"source": "kmnet_executor", **buttons},
                )
                self._log_box_input_state(hardware_state, "hardware")
                return hardware_state
        reason = str((buttons or {}).get("reason") or "kmNet button reader unavailable")
        state = BoxInputState(raw={"source": "kmnet_executor", "reason": reason})
        self._log_box_input_state(state, "kmnet_unavailable")
        return state

    def _create_control_strategy(self, config: RuntimeConfig):
        if config.control.strategy == "experimental_angle_pid":
            return ExperimentalAnglePidStrategy(
                kp_x=config.control.experimental_angle_kp_x,
                kp_y=config.control.experimental_angle_kp_y,
                ki=config.control.experimental_angle_ki,
                kd=config.control.experimental_angle_kd,
                integral_limit=config.control.experimental_angle_integral_limit,
                speed=config.control.experimental_angle_speed,
                smooth_factor=config.control.experimental_angle_smooth_factor,
                deadzone_px=config.control.experimental_angle_deadzone_px,
                derivative_filter=config.control.experimental_angle_derivative_filter,
                near_error_deg=config.control.experimental_angle_near_error_deg,
                far_error_deg=config.control.experimental_angle_far_error_deg,
                near_kp_scale=config.control.experimental_angle_near_kp_scale,
                middle_kp_scale=config.control.experimental_angle_middle_kp_scale,
                far_kp_scale=config.control.experimental_angle_far_kp_scale,
                near_kd_scale=config.control.experimental_angle_near_kd_scale,
                middle_kd_scale=config.control.experimental_angle_middle_kd_scale,
                far_kd_scale=config.control.experimental_angle_far_kd_scale,
                prediction_gain_min=config.control.experimental_angle_prediction_gain_min,
                prediction_d_gain_min=config.control.experimental_angle_prediction_d_gain_min,
                max_control_angle_deg=config.control.experimental_angle_max_control_angle_deg,
                calibration_profile_id=config.calibration.profile_id,
                calibration_profile_version=config.calibration.profile_version,
                fov_semantics=config.calibration.fov_semantics,
                fov_x_deg=config.calibration.fov_x_deg,
                counts_per_360_x=config.calibration.counts_per_360_x,
                counts_per_360_y=config.calibration.counts_per_360_y,
                axis_sign_x=config.calibration.axis_sign_x,
                axis_sign_y=config.calibration.axis_sign_y,
                game_sensitivity_fingerprint=config.calibration.game_sensitivity_fingerprint,
                projection_profile=config.calibration.projection_profile,
                max_step_counts=config.control.experimental_angle_max_step_counts,
                max_counts_delta_x=config.control.experimental_angle_max_counts_delta_x,
                max_counts_delta_y=config.control.experimental_angle_max_counts_delta_y,
                control_hz=config.control.experimental_angle_control_hz,
                kalman_enabled=config.control.experimental_angle_kalman_enabled,
                kalman_process_noise=config.control.experimental_angle_kalman_process_noise,
                kalman_measurement_noise=config.control.experimental_angle_kalman_measurement_noise,
                hungarian_enabled=config.control.experimental_angle_hungarian_enabled,
                matching_distance_px=config.control.experimental_angle_matching_distance_px,
                max_extrapolate_frames=config.control.experimental_angle_max_extrapolate_frames,
                target_filter_enabled=config.control.experimental_angle_target_filter_enabled,
                target_filter_min_score=config.control.experimental_angle_target_filter_min_score,
                target_filter_fov_ratio=config.control.experimental_angle_target_filter_fov_ratio,
                target_filter_same_class=config.control.experimental_angle_target_filter_same_class,
                prediction_lead_ms=config.control.experimental_angle_prediction_lead_ms,
                extrapolate_confidence_decay=config.control.experimental_angle_extrapolate_confidence_decay,
                magnet_enabled=config.control.experimental_angle_magnet_enabled,
                magnet_radius_px=config.control.experimental_angle_magnet_radius_px,
                magnet_strength=config.control.experimental_angle_magnet_strength,
                magnet_curve=config.control.experimental_angle_magnet_curve,
                magnet_deadzone_px=config.control.experimental_angle_magnet_deadzone_px,
                magnet_max_counts=config.control.experimental_angle_magnet_max_counts,
                capture_width=config.capture.width,
                capture_height=config.capture.height,
                move_kind=config.control.move_kind,
                move_ms=config.control.move_ms,
                trace_ms=config.control.trace_ms,
                bezier_curvature=config.control.bezier_curvature,
            )
        raise ValueError("control.strategy must be experimental_angle_pid")

    def _reset_control_motion_state(self) -> None:
        reset_strategy = getattr(self.control_strategy, "reset", None)
        if callable(reset_strategy):
            reset_strategy()
            return
        reset = getattr(self.control_strategy, "_reset_motion_state", None)
        if callable(reset):
            reset()
            return
        for name, value in (
            ("_last_center", None),
            ("_ema_x", 0.0),
            ("_ema_y", 0.0),
        ):
            if hasattr(self.control_strategy, name):
                setattr(self.control_strategy, name, value)

    @staticmethod
    def _target_detection_index(context: FrameContext, target: Track | Detection) -> int | None:
        for index, detection in enumerate(context.detections):
            if detection is target:
                return index
        for index, detection in enumerate(context.detections):
            if (
                int(detection.cls) == int(target.cls)
                and abs(float(detection.x) - float(target.x)) <= 1e-3
                and abs(float(detection.y) - float(target.y)) <= 1e-3
                and abs(float(detection.w) - float(target.w)) <= 1e-3
                and abs(float(detection.h) - float(target.h)) <= 1e-3
            ):
                return index
        return None

    def _control_target_key(self, target: Track | Detection, context: FrameContext) -> str:
        track_id = getattr(target, "track_id", None)
        if track_id is not None:
            return f"track:{int(track_id)}"
        detection_index = self._target_detection_index(context, target)
        if detection_index is not None:
            return f"det:{detection_index}:class:{int(target.cls)}"
        return f"class:{int(target.cls)}"

    @staticmethod
    def _with_strategy_target(
        state: BoxInputState,
        target_key: str,
        *,
        frame_age_ms: float,
        frame_id: int,
        capture_ts_ns: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> BoxInputState:
        return BoxInputState(
            left=state.left,
            right=state.right,
            side=state.side,
            raw={
                **(state.raw or {}),
                "target_key": target_key,
                "frame_age_ms": frame_age_ms,
                "frame_id": frame_id,
                "capture_ts_ns": capture_ts_ns,
                **(metadata or {}),
            },
        )

    def _aim_latency_metadata(
        self,
        *,
        context: FrameContext,
        target: Track | Detection,
        strategy_metadata: dict[str, Any],
        selector_debug: dict[str, Any],
        compute_ts_ns: int,
    ) -> dict[str, Any]:
        if not isinstance(target, Track):
            return {}
        control = self.config.control
        aim_cfg = AimPointConfig(
            horizontal_percent=float(getattr(control, "aim_horizontal_percent", 50.0)),
            vertical_percent_from_top=self._active_aim_ratio(),
            offset_x_px=float(getattr(control, "aim_offset_x_px", 0.0)),
            offset_y_px=float(getattr(control, "aim_offset_y_px", 0.0)),
            ema_enabled=bool(getattr(control, "aim_ema_enabled", True)),
            ema_alpha=float(getattr(control, "aim_ema_alpha", 0.65)),
            max_anchor_jump_ratio=float(getattr(control, "aim_max_anchor_jump_ratio", 0.15)),
        )
        latency_cfg = LatencyCompensationConfig(
            enabled=bool(getattr(control, "latency_compensation_enabled", True)),
            scale=float(getattr(control, "latency_compensation_scale", 0.70)),
            max_compensation_ms=float(getattr(control, "latency_max_compensation_ms", 35.0)),
            reject_if_age_exceeds_ms=float(getattr(control, "latency_reject_if_age_exceeds_ms", 55.0)),
            max_compensation_px=float(getattr(control, "latency_max_compensation_px", 80.0)),
            min_velocity_px_s=float(getattr(control, "latency_min_velocity_px_s", 30.0)),
            max_velocity_px_s=float(getattr(control, "latency_max_velocity_px_s", 2500.0)),
            min_velocity_measurements=int(getattr(control, "latency_min_velocity_measurements", 3)),
            min_velocity_confidence=float(getattr(control, "latency_min_velocity_confidence", 0.65)),
            estimated_actuation_delay_ms=float(getattr(control, "latency_estimated_actuation_delay_ms", 2.0)),
        )
        estimate = estimated_state_from_debug(
            track=target,
            capture_ts_ns=context.capture_ts_ns,
            tracker_debug=selector_debug,
        )
        aim = self.aim_points.update(track=target, estimate=estimate, config=aim_cfg)
        compensated = self.latency_compensator.compensate(
            aim=aim,
            estimate=estimate,
            source_frame_id=context.frame_id,
            compute_ts_ns=compute_ts_ns,
            roi_offset_x=float(strategy_metadata.get("roi_offset_x") or 0.0),
            roi_offset_y=float(strategy_metadata.get("roi_offset_y") or 0.0),
            roi_width=float(strategy_metadata.get("roi_width") or context.width),
            roi_height=float(strategy_metadata.get("roi_height") or context.height),
            control_width=float(strategy_metadata.get("control_width") or 0.0),
            control_height=float(strategy_metadata.get("control_height") or 0.0),
            config=latency_cfg,
            coordinate_transform=self._coordinate_transform_for_context(context),
        )
        aim_payload = {
            "track_id": aim.track_id,
            "state_ts_ns": aim.state_ts_ns,
            "raw_x": aim.raw_x,
            "raw_y": aim.raw_y,
            "smoothed_x": aim.smoothed_x,
            "smoothed_y": aim.smoothed_y,
            "center_x": aim.center_x,
            "center_y": aim.center_y,
            "anchor_jump_norm": aim.anchor_jump_norm,
            "aim_confidence": aim.aim_confidence,
            "ema_reset_reason": aim.ema_reset_reason,
            "horizontal_percent": aim_cfg.horizontal_percent,
            "vertical_percent_from_top": aim_cfg.vertical_percent_from_top,
            "offset_x_px": aim_cfg.offset_x_px,
            "offset_y_px": aim_cfg.offset_y_px,
            "ema_enabled": aim_cfg.ema_enabled,
            "ema_alpha": aim_cfg.ema_alpha,
            "max_anchor_jump_ratio": aim_cfg.max_anchor_jump_ratio,
        }
        estimated_payload = {
            "track_id": estimate.track_id,
            "state_ts_ns": estimate.state_ts_ns,
            "capture_ts_ns": estimate.capture_ts_ns,
            "x": estimate.x,
            "y": estimate.y,
            "vx": estimate.vx,
            "vy": estimate.vy,
            "valid": estimate.valid,
            "predicted": estimate.predicted,
            "prediction_confidence": estimate.prediction_confidence,
            "position_sigma_px": estimate.position_sigma_px,
            "cov_trace": estimate.cov_trace,
            "nis": estimate.nis,
            "identity_confidence": estimate.identity_confidence,
            "velocity_measurements": estimate.velocity_measurements,
        }
        return {
            "aim_point": aim_payload,
            "estimated_target_state": estimated_payload,
            "compensated_target": compensated.debug_payload(),
            "latency_compensation": {
                "enabled": latency_cfg.enabled,
                "scale": latency_cfg.scale,
                "max_compensation_ms": latency_cfg.max_compensation_ms,
                "reject_if_age_exceeds_ms": latency_cfg.reject_if_age_exceeds_ms,
                "max_compensation_px": latency_cfg.max_compensation_px,
                "min_velocity_px_s": latency_cfg.min_velocity_px_s,
                "max_velocity_px_s": latency_cfg.max_velocity_px_s,
                "min_velocity_measurements": latency_cfg.min_velocity_measurements,
                "min_velocity_confidence": latency_cfg.min_velocity_confidence,
                "estimated_actuation_delay_ms": latency_cfg.estimated_actuation_delay_ms,
            },
        }

    def _strategy_frame_metadata(self, context: FrameContext) -> dict[str, Any]:
        inference_debug = self.last_inference_status.get("debug", {})
        preprocess = inference_debug.get("preprocess", {}) if isinstance(inference_debug, dict) else {}
        source_width = self._status_int("source_width", 0)
        source_height = self._status_int("source_height", 0)
        source_geometry_trusted = self.last_inference_status.get("source_geometry_trusted")
        if source_geometry_trusted is False:
            source_width = 0
            source_height = 0
        roi_offset_x = self._status_int("roi_offset_x", 0)
        roi_offset_y = self._status_int("roi_offset_y", 0)
        return {
            "roi_width": context.width,
            "roi_height": context.height,
            "capture_width": source_width,
            "capture_height": source_height,
            "capture_geometry_source": self.last_inference_status.get("source_geometry_source", ""),
            "capture_geometry_trusted": source_geometry_trusted,
            "control_width": source_width,
            "control_height": source_height,
            "roi_offset_x": roi_offset_x,
            "roi_offset_y": roi_offset_y,
            "model_width": preprocess.get("model_width") if isinstance(preprocess, dict) else None,
            "model_height": preprocess.get("model_height") if isinstance(preprocess, dict) else None,
            "detections": [
                {
                    "index": index,
                    "cls": int(detection.cls),
                    "score": float(detection.score),
                    "x1": float(detection.x1),
                    "y1": float(detection.y1),
                    "x2": float(detection.x2),
                    "y2": float(detection.y2),
                }
                for index, detection in enumerate(context.detections)
            ],
        }

    def _status_int(self, key: str, fallback: int) -> int:
        value = self.last_inference_status.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        return int(fallback)

    @classmethod
    def _candidate_filter_payload(cls, selector_debug: dict[str, Any]) -> dict[str, Any]:
        candidates = selector_debug.get("candidates")
        rejected = selector_debug.get("rejected")
        selected = selector_debug.get("selected")
        return {
            "raw_candidates": cls._debug_int(selector_debug, "raw_candidates"),
            "filtered_candidates": cls._debug_int(selector_debug, "filtered_candidates"),
            "inside_fov": cls._debug_int(selector_debug, "inside_fov"),
            "rejected_candidates": cls._debug_int(selector_debug, "rejected_candidates"),
            "candidates": cls._debug_dict_list(candidates),
            "rejected": cls._debug_dict_list(rejected),
            "selected": dict(selected) if isinstance(selected, dict) else None,
            "reason": str(selector_debug.get("reason") or ""),
        }

    def _track_diagnostics_payload(
        self,
        selector_debug: dict[str, Any],
        *,
        target: Track | Detection | None = None,
        selection: TargetSelection | None = None,
    ) -> dict[str, Any]:
        tracker = selector_debug.get("tracker") if isinstance(selector_debug, dict) else {}
        if not isinstance(tracker, dict):
            tracker = {}
        switch = selector_debug.get("switch") if isinstance(selector_debug, dict) else {}
        if not isinstance(switch, dict):
            switch = {}
        tracks = self._debug_dict_list(tracker.get("tracks"))
        selected_track_id = int(target.track_id) if isinstance(target, Track) else None
        selected_track = self._find_track_debug(tracks, selected_track_id)
        selected_continuity = self._optional_float(selector_debug.get("selected_continuity_score"))
        if selected_continuity is None and selected_track is not None:
            selected_continuity = self._optional_float(selected_track.get("identity_confidence"))
        selection_state = str(selection.state) if selection is not None else ""
        selection_reason = str(selection.reason) if selection is not None else ""
        switch_reason = str(
            switch.get("reason")
            or selection_reason
            or tracker.get("reason")
            or selector_debug.get("reason")
            or ""
        )
        selected_state = (
            str(selected_track.get("state") or "")
            if selected_track is not None
            else selection_state if selected_track_id is not None else ""
        )
        return {
            "tracker_state": str(tracker.get("state") or selection_state or ""),
            "tracker_reason": str(tracker.get("reason") or selector_debug.get("reason") or ""),
            "selection_state": selection_state,
            "selection_reason": selection_reason,
            "selected_track_id": selected_track_id,
            "selected_track_state": selected_state,
            "selected_continuity_score": selected_continuity,
            "selected_identity_confidence": self._optional_float(
                selected_track.get("identity_confidence") if selected_track is not None else None
            ),
            "selected_missing_ms": self._optional_float(
                selected_track.get("missing_ms") if selected_track is not None else None
            ),
            "selected_hits": self._optional_int(
                selected_track.get("hits") if selected_track is not None else None
            ),
            "selected_misses": self._optional_int(
                selected_track.get("misses") if selected_track is not None else None
            ),
            "selected_last_match_cost": self._optional_float(
                selected_track.get("last_match_cost") if selected_track is not None else None
            ),
            "selected_mahalanobis": self._optional_float(
                selected_track.get("mahalanobis") if selected_track is not None else None
            ),
            "switch_state": str(switch.get("state") or ""),
            "switch_committed": bool(switch.get("committed", False)),
            "switch_reason": switch_reason,
            "pending_switch": dict(switch) if switch.get("state") == "pending" else None,
            "identity_uncertain_tracks": self._debug_int_list(tracker.get("identity_uncertain_tracks")),
            "available_tracks": self._optional_int(tracker.get("available_tracks")),
            "track_count": len(tracks),
            "tracks": tracks,
        }

    @staticmethod
    def _debug_dict_list(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    @staticmethod
    def _find_track_debug(tracks: list[dict[str, Any]], track_id: int | None) -> dict[str, Any] | None:
        if track_id is None:
            return None
        for item in tracks:
            try:
                if int(item.get("track_id")) == track_id:
                    return item
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _debug_int_list(value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if isinstance(value, (int, float)):
            return int(value)
        return None

    @staticmethod
    def _frame_age_ms(context: FrameContext) -> float:
        if context.capture_ts_ns is None:
            return 0.0
        return max(0.0, (time.monotonic_ns() - int(context.capture_ts_ns)) / 1e6)

    def _target_payload(self, target: Track | Detection, context: FrameContext) -> dict[str, Any]:
        class_name = self._class_display_name(int(target.cls), context)
        coordinate_payload = self._coordinate_payload(target, context)
        payload = {
            "frame_id": context.frame_id,
            "class_id": int(target.cls),
            "class_name": class_name,
            "score": float(target.score),
            "x": float(target.x),
            "y": float(target.y),
            "w": float(target.w),
            "h": float(target.h),
            "x1": float(target.x1),
            "y1": float(target.y1),
            "x2": float(target.x2),
            "y2": float(target.y2),
            "cx": float(target.cx),
            "cy": float(target.cy),
            "box_cx": float(target.cx),
            "box_cy": float(target.cy),
            "offset_x": float(target.cx - context.width / 2),
            "offset_y": float(target.cy - context.height / 2),
            "box_offset_x": float(target.cx - context.width / 2),
            "box_offset_y": float(target.cy - context.height / 2),
            **coordinate_payload,
        }
        if isinstance(target, Track):
            payload["track_id"] = int(target.track_id)
        return payload

    def _detection_payload(self, detection: Detection, context: FrameContext) -> dict[str, Any]:
        class_name = self._class_display_name(int(detection.cls), context)
        coordinate_payload = self._coordinate_payload(detection, context)
        return {
            "frame_id": context.frame_id,
            "class_id": int(detection.cls),
            "class_name": class_name,
            "score": float(detection.score),
            "x": float(detection.x),
            "y": float(detection.y),
            "w": float(detection.w),
            "h": float(detection.h),
            "x1": float(detection.x1),
            "y1": float(detection.y1),
            "x2": float(detection.x2),
            "y2": float(detection.y2),
            "cx": float(detection.cx),
            "cy": float(detection.cy),
            **coordinate_payload,
        }

    def _coordinate_payload(self, item: Track | Detection, context: FrameContext) -> dict[str, Any]:
        roi_box = self._box_payload(
            x=float(item.x),
            y=float(item.y),
            w=float(item.w),
            h=float(item.h),
        )
        transform = self._coordinate_transform_for_context(context)
        trusted = transform is not None
        roi_offset_x = int(transform.roi_x) if transform is not None else self._status_int("roi_offset_x", 0)
        roi_offset_y = int(transform.roi_y) if transform is not None else self._status_int("roi_offset_y", 0)
        source_width = int(transform.capture_width) if transform is not None else 0
        source_height = int(transform.capture_height) if transform is not None else 0
        capture_box = None
        control_box = None
        full_display_box = None
        if transform is not None:
            capture_box = self._bbox_payload(transform.roi_to_capture_box(item.box))
            control_box = self._bbox_payload(transform.roi_to_control_box(item.box))
            full_display_box = self._bbox_payload(transform.roi_to_display_box(item.box))
        return {
            "coordinate_space": "roi",
            "coordinate_spaces": {
                "native": "roi",
                "display": "roi_preview",
                "capture": "capture" if trusted else "",
                "control": "control" if trusted else "",
                "full_display": "display" if trusted else "",
            },
            "roi_box": roi_box,
            "display_box": dict(roi_box),
            "capture_box": capture_box,
            "control_box": control_box,
            "full_display_box": full_display_box,
            "roi_offset_x": roi_offset_x,
            "roi_offset_y": roi_offset_y,
            "source_width": source_width if trusted else 0,
            "source_height": source_height if trusted else 0,
            "source_geometry_trusted": trusted,
            "source_geometry_source": self.last_inference_status.get("source_geometry_source", ""),
            "roi_width": int(context.width),
            "roi_height": int(context.height),
            "coordinate_transform": self._coordinate_transform_payload(transform),
        }

    @staticmethod
    def _box_payload(*, x: float, y: float, w: float, h: float) -> dict[str, float]:
        return {
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "x1": float(x),
            "y1": float(y),
            "x2": float(x) + float(w),
            "y2": float(y) + float(h),
            "cx": float(x) + float(w) / 2.0,
            "cy": float(y) + float(h) / 2.0,
        }

    @classmethod
    def _bbox_payload(cls, box: BBox) -> dict[str, float]:
        return cls._box_payload(x=box.x1, y=box.y1, w=box.width, h=box.height)

    def _coordinate_transform_for_context(self, context: FrameContext) -> CoordinateTransform | None:
        if self.last_inference_status.get("source_geometry_trusted") is not True:
            return None
        source_width = self._status_int("source_width", 0)
        source_height = self._status_int("source_height", 0)
        if source_width <= 0 or source_height <= 0 or context.width <= 0 or context.height <= 0:
            return None
        try:
            return CoordinateTransform(
                model_width=float(context.width),
                model_height=float(context.height),
                roi_x=float(self._status_int("roi_offset_x", 0)),
                roi_y=float(self._status_int("roi_offset_y", 0)),
                roi_width=float(context.width),
                roi_height=float(context.height),
                capture_width=float(source_width),
                capture_height=float(source_height),
            )
        except ValueError:
            return None

    @staticmethod
    def _coordinate_transform_payload(transform: CoordinateTransform | None) -> dict[str, float] | None:
        if transform is None:
            return None
        return {
            "model_width": float(transform.model_width),
            "model_height": float(transform.model_height),
            "roi_x": float(transform.roi_x),
            "roi_y": float(transform.roi_y),
            "roi_width": float(transform.roi_width),
            "roi_height": float(transform.roi_height),
            "capture_width": float(transform.capture_width),
            "capture_height": float(transform.capture_height),
            "control_origin_x": float(transform.control_origin_x),
            "control_origin_y": float(transform.control_origin_y),
            "display_scale_x": float(transform.display_scale_x),
            "display_scale_y": float(transform.display_scale_y),
        }

    def _vision_status(self) -> dict[str, Any]:
        context = self.last_frame_context
        if context is None:
            return {
                "frame_id": None,
                "detections": 0,
                "detection_items": [],
                "inference_reason": self.last_inference_reason,
                "inference": dict(self.last_inference_status),
                "calibration": self._calibration_fingerprint_status(),
                "target": None,
                "control": None,
                "execution": self.last_execution,
                "trace": self._business_trace(None),
            }
        return {
            "frame_id": context.frame_id,
            "detections": len(context.detections),
            "detection_items": [
                self._detection_payload(item, context)
                for item in context.detections
            ],
            "tracks": len(context.tracks),
            "classes": list(context.classes),
            "inference_reason": self.last_inference_reason,
            "inference": dict(self.last_inference_status),
            "calibration": self._calibration_fingerprint_status(),
            "target": self.last_target,
            "control": self.last_control,
            "execution": self.last_execution,
            "trace": self._business_trace(context),
        }

    def _business_trace(self, context: FrameContext | None) -> dict[str, Any]:
        capture_state = getattr(getattr(self, "capture", None), "state", None)
        inference = dict(self.last_inference_status)
        control = dict(self.last_control or {})
        execution = dict(self.last_execution or {})
        stages = [
            self._trace_capture_stage(capture_state),
            self._trace_roi_stage(inference),
            self._trace_inference_stage(inference),
            self._trace_target_stage(context, control),
            self._trace_control_stage(control),
            self._trace_execution_stage(execution),
        ]
        failed = next((item for item in stages if item["status"] == "failed"), None)
        blocked = next((item for item in stages if item["status"] == "blocked"), None)
        summary = failed or blocked or stages[-1]
        return {
            "frame_id": getattr(context, "frame_id", None),
            "status": summary["status"],
            "blocked_at": summary["id"] if summary["status"] in {"failed", "blocked"} else "",
            "message": summary["message"],
            "stages": stages,
        }

    @staticmethod
    def _trace_capture_stage(capture_state: Any) -> dict[str, Any]:
        available = getattr(capture_state, "available", False) is True
        message = "采集正常"
        status = "ok" if available else "blocked"
        if not available:
            message = str(getattr(capture_state, "last_error", "") or "采集未启动")
        profile = getattr(capture_state, "profile", None)
        detail = ""
        if profile is not None:
            detail = f"{getattr(profile, 'pixel_format', '-')}/{getattr(profile, 'width', 0)}x{getattr(profile, 'height', 0)}@{getattr(profile, 'fps', 0)}"
        return {"id": "capture", "label": "采集", "status": status, "message": message, "detail": detail}

    @staticmethod
    def _trace_roi_stage(inference: dict[str, Any]) -> dict[str, Any]:
        if not inference:
            return {"id": "roi", "label": "ROI", "status": "blocked", "message": "等待采集帧", "detail": ""}
        width = inference.get("input_width")
        height = inference.get("input_height")
        detail = f"{width}x{height}" if width and height else ""
        return {"id": "roi", "label": "ROI", "status": "ok", "message": "ROI 帧已生成", "detail": detail}

    @staticmethod
    def _trace_inference_stage(inference: dict[str, Any]) -> dict[str, Any]:
        if not inference.get("ran"):
            return {"id": "inference", "label": "推理", "status": "blocked", "message": inference.get("reason") or "推理尚未执行", "detail": ""}
        if inference.get("available") is not True:
            return {"id": "inference", "label": "推理", "status": "failed", "message": inference.get("reason") or "推理失败", "detail": ""}
        mapped = int(inference.get("mapped_detections") or 0)
        if mapped <= 0:
            raw = int(inference.get("raw_detections") or 0)
            return {"id": "inference", "label": "推理", "status": "blocked", "message": "未检测到可用目标", "detail": f"raw={raw}, mapped={mapped}"}
        return {"id": "inference", "label": "推理", "status": "ok", "message": "推理有检测结果", "detail": f"mapped={mapped}"}

    def _trace_target_stage(self, context: FrameContext | None, control: dict[str, Any]) -> dict[str, Any]:
        if context is None:
            return {"id": "target", "label": "目标", "status": "blocked", "message": "等待推理帧", "detail": ""}
        detections = len(context.detections)
        if self.last_target is None:
            return {
                "id": "target",
                "label": "目标",
                "status": "blocked",
                "message": str(control.get("selection_reason") or "没有选中目标"),
                "detail": f"candidates={detections}",
            }
        target = self.last_target
        return {
            "id": "target",
            "label": "目标",
            "status": "ok",
            "message": str(target.get("class_name") or "目标已选中"),
            "detail": f"idx={target.get('target_detection_index', '-')}, score={float(target.get('score') or 0):.2f}",
        }

    @staticmethod
    def _trace_control_stage(control: dict[str, Any]) -> dict[str, Any]:
        if not control:
            return {"id": "control", "label": "控制量", "status": "blocked", "message": "没有目标，未计算控制量", "detail": ""}
        dx = float(control.get("dx") or 0.0)
        dy = float(control.get("dy") or 0.0)
        frame_age_ms = float(control.get("frame_age_ms") or 0.0)
        detail = f"dx={dx:.1f}, dy={dy:.1f}, age={frame_age_ms:.1f}ms"
        if control.get("will_emit") is not True:
            calibration = control.get("calibration_status")
            if isinstance(calibration, dict) and calibration.get("control_allowed") is False:
                message = str(calibration.get("reason") or "Calibration fingerprint mismatch")
                return {
                    "id": "control",
                    "label": "控制量",
                    "status": "blocked",
                    "message": message,
                    "detail": detail,
                }
            return {
                "id": "control",
                "label": "控制量",
                "status": "blocked",
                "message": str(control.get("trigger_reason") or "等待触发，控制量未发送"),
                "detail": detail,
            }
        if round(dx) == 0 and round(dy) == 0:
            return {"id": "control", "label": "控制量", "status": "blocked", "message": str(control.get("reason") or "控制量为 0"), "detail": detail}
        return {"id": "control", "label": "控制量", "status": "ok", "message": str(control.get("reason") or "控制量已生成"), "detail": detail}

    @staticmethod
    def _trace_execution_stage(execution: dict[str, Any]) -> dict[str, Any]:
        if not execution:
            return {"id": "execution", "label": "执行", "status": "blocked", "message": "没有控制命令", "detail": ""}
        dx = float(execution.get("output_dx") or 0.0)
        dy = float(execution.get("output_dy") or 0.0)
        detail = f"{execution.get('executor_id') or ''} dx={dx:.0f}, dy={dy:.0f}"
        if execution.get("sent") is True:
            return {"id": "execution", "label": "执行", "status": "ok", "message": str(execution.get("message") or "已发送"), "detail": detail}
        message = str(execution.get("message") or "未发送")
        status = "failed" if "failed" in message.lower() or "unavailable" in message.lower() else "blocked"
        return {"id": "execution", "label": "执行", "status": status, "message": message, "detail": detail}

    def _record_control_frame(self) -> None:
        if not bool(getattr(getattr(self.config, "consumers", None), "recording", False)):
            return
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            return
        try:
            record = build_control_frame_record(
                control=self.last_control,
                target=self.last_target,
                inference=self.last_inference_status,
                execution=self.last_execution,
                config=self.config,
                scheduler_status=self._scheduler_status(),
            )
            writer = getattr(recorder, "record_control_frame", None)
            if callable(writer):
                writer(record)
                return
            writer = getattr(recorder, "record", None)
            if callable(writer):
                writer(record)
        except Exception as exc:
            logger.warning("control frame recorder failed: %s", exc)

    def _scheduler_status(self) -> dict[str, Any]:
        scheduler = getattr(self.executors, "scheduler", None)
        status = getattr(scheduler, "status", None)
        if not callable(status):
            return {}
        try:
            payload = status()
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _execution_result_payload(self, result: Any) -> dict[str, Any]:
        intent = getattr(result, "intent", None)
        metadata = getattr(result, "metadata", None) or {}
        return {
            "executor_id": str(getattr(result, "executor_id", "")),
            "sent": bool(getattr(result, "sent", False)),
            "message": str(getattr(result, "message", "")),
            "metadata": dict(metadata),
            "accepted": bool(getattr(intent, "accepted", False)),
            "clipped": bool(getattr(intent, "clipped", False)),
            "output_dx": float(getattr(intent, "dx", 0.0)),
            "output_dy": float(getattr(intent, "dy", 0.0)),
            "move_kind": str(getattr(intent, "move_kind", "")),
            "move_ms": int(getattr(intent, "move_ms", 0)),
            "intent": {
                "dx": float(getattr(intent, "dx", 0.0)),
                "dy": float(getattr(intent, "dy", 0.0)),
                "accepted": bool(getattr(intent, "accepted", False)),
                "clipped": bool(getattr(intent, "clipped", False)),
                "reason": str(getattr(intent, "reason", "")),
            },
        }

    def _attach_execution_to_last_control(self, execution: dict[str, Any]) -> None:
        if not isinstance(self.last_control, dict):
            return
        metadata = execution.get("metadata") if isinstance(execution, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        driver_dx = metadata.get("driver_dx")
        driver_dy = metadata.get("driver_dy")
        driver_counts = None
        if isinstance(driver_dx, (int, float)) and isinstance(driver_dy, (int, float)):
            driver_counts = {"dx": float(driver_dx), "dy": float(driver_dy)}
        payload = {
                "execution": execution,
                "execution_sent": bool(execution.get("sent", False)),
                "execution_executor": str(execution.get("executor_id", "")),
                "execution_message": str(execution.get("message", "")),
                "driver_api": str(metadata.get("api_name") or ""),
                "driver_rc": metadata.get("driver_rc"),
                "driver_dx": float(driver_dx) if isinstance(driver_dx, (int, float)) else None,
                "driver_dy": float(driver_dy) if isinstance(driver_dy, (int, float)) else None,
                "driver_counts": driver_counts,
        }
        execution_global_state = self._global_state_from_execution_metadata(metadata)
        if execution_global_state is not None:
            payload["global_state"] = execution_global_state
        self.last_control.update(payload)

    def _source_width(self, frame: CapturedFrame) -> int:
        return int(frame.source_width or frame.width)

    def _source_height(self, frame: CapturedFrame) -> int:
        return int(frame.source_height or frame.height)

    @staticmethod
    def _capture_geometry(frame: CapturedFrame) -> tuple[int, int, str, bool]:
        source_width = getattr(frame, "source_width", None)
        source_height = getattr(frame, "source_height", None)
        if isinstance(source_width, int) and isinstance(source_height, int) and source_width > 0 and source_height > 0:
            return source_width, source_height, "source_metadata", True

        roi_size = getattr(frame, "roi_size", None)
        roi_offset_x = int(getattr(frame, "roi_offset_x", 0) or 0)
        roi_offset_y = int(getattr(frame, "roi_offset_y", 0) or 0)
        if roi_size is not None or roi_offset_x != 0 or roi_offset_y != 0:
            return 0, 0, "missing_source_geometry", False

        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        if width > 0 and height > 0:
            return width, height, "frame_size", True
        return 0, 0, "missing_source_geometry", False

    def _class_display_name(self, class_id: int, context: FrameContext) -> str:
        profiles = getattr(self.config.inference, "detection_class_profiles", {}) or {}
        profile_name = str(getattr(self.config.inference, "detection_class_profile", "default"))
        profile = profiles.get(profile_name) or profiles.get("default") or []
        if 0 <= class_id < len(profile):
            return str(profile[class_id])
        if 0 <= class_id < len(context.classes):
            return context.classes[class_id]
        return str(class_id)

    def _image_width(self, image: Any | None) -> int | None:
        if image is None:
            return None
        if hasattr(image, "shape"):
            try:
                return int(image.shape[1])
            except Exception:
                return None
        return int(getattr(image, "width")) if getattr(image, "width", None) is not None else None

    def _image_height(self, image: Any | None) -> int | None:
        if image is None:
            return None
        if hasattr(image, "shape"):
            try:
                return int(image.shape[0])
            except Exception:
                return None
        return int(getattr(image, "height")) if getattr(image, "height", None) is not None else None

    def _active_model(self) -> dict | None:
        deployment = self.models.get_active_deployment()
        if deployment is None:
            return None

        project = self.models.get_project(deployment.project_id)
        artifact = self.models.get_artifact(deployment.artifact_id)
        version = self.models.get_version(artifact.version_id) if artifact is not None else None
        return {
            "project": asdict(project) if project is not None else None,
            "version": asdict(version) if version is not None else None,
            "deployment": asdict(deployment),
            "artifact": asdict(artifact) if artifact is not None else None,
        }
