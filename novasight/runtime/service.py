from __future__ import annotations

from collections import deque
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
    CALIBRATED_ANGULAR,
    DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2,
    UNIVERSAL_SATURATED,
    AlgorithmRegistry,
    CalibratedAngularControllerConfig,
    MoveCommand,
    MouseControllerConfig,
    MouseController,
    MouseObservation,
    RawAimPointProjector,
    SharedOutputConfig,
    UniversalSaturatedControllerConfig,
    plan_step_capacity,
    resolve_aim_y_ratio,
    target_motion_estimate_from_debug,
)
from novasight.control.algorithms.dual_phase_atan_robust_predictive_v2 import (
    AtanControllerConfig as DualPhaseRobustAtanControllerConfig,
    AtanModeConfig as DualPhaseRobustAtanModeConfig,
    DualPhaseAtanRobustPredictiveV2Algorithm,
    DualPhaseAtanRobustPredictiveV2Config as DualPhaseRobustAlgorithmConfig,
    DualPhaseAtanRobustPredictiveV2Observation,
    ModeSelectorConfig as DualPhaseRobustModeSelectorConfig,
    PredictionConfig as DualPhaseRobustPredictionConfig,
    PredictionModeConfig as DualPhaseRobustPredictionModeConfig,
    ProjectionConfig as DualPhaseRobustProjectionConfig,
    RecoilConfig as DualPhaseRobustRecoilConfig,
    VelocityConfig as DualPhaseRobustVelocityConfig,
)
from novasight.executors import BoxInputState, ExecutorRegistry
from novasight.inference import InferenceResult
from novasight.inference.jetson import create_gpu_resource_preprocessor
from novasight.model_registry import ModelRegistry
from novasight.contracts import BBox, ControlIntent, Detection, DetectionBatch, FrameContext, Track
from novasight.roi import center_roi_frame, center_roi_region

from .candidates import parse_allowed_class_ids
from .config_store import RuntimeConfigStore
from .control_timing import ControlTimingModel
from .control_trace import build_control_trace_record
from .detection_batch import detection_batch_to_frame_context
from .freshness import FreshnessGate
from .recorder import build_control_frame_record
from .state import RuntimeFrameResult, RuntimeState
from .target_selector import RuntimeTargetSelector, TargetSelection

logger = logging.getLogger("novasight.runtime.service")


def _status_statistic(
    status: dict[str, Any],
    section: str,
    key: str,
) -> float:
    values = status.get(section, {})
    if not isinstance(values, dict):
        return 0.0
    try:
        return float(values.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


class RuntimeService:
    def __init__(
        self,
        config: RuntimeConfig,
        models: ModelRegistry,
        executors: ExecutorRegistry,
        capture: Any | None = None,
        inference: Any | None = None,
        recorder: Any | None = None,
    ) -> None:
        self.config = config
        self.models = models
        self.executors = executors
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
        self._last_box_input_log_signature = ""
        self._last_no_target_log_signature = ""
        self._control_lock = threading.Lock()
        self._last_control_tick_ns = 0
        self._executed_control_samples: deque[tuple[int, int, int]] = deque()
        self.control_timing = ControlTimingModel()
        self.last_control_timing: dict[str, Any] = {}
        self.control_algorithms = self._create_control_algorithm_registry(config)
        self.target_selector = RuntimeTargetSelector()
        self.raw_aim_projector = RawAimPointProjector()
        self._runtime_calibration_signature = self._config_calibration_signature(config)
        self._external_sensitivity_fingerprint = ""
        self._external_sensitivity_source = ""
        self._external_sensitivity_ts_ns = 0
        self._last_runtime_reset_reason = ""
        self._accepted_batch_generation = -1
        self._accepted_batch_capture_ts_ns = 0
        self.stale_drop_count = 0
        self._log_production_control_chain("startup")

    def state(self) -> RuntimeState:
        with self._control_lock:
            inference_observation = dict(self.last_inference_status)
            pipeline_timings = dict(self.last_pipeline_timings)
            vision = self._vision_status()
        capture_state = getattr(self, "capture", None)
        inference_state = getattr(self, "inference", None)
        active_model = self._active_model()
        capture_payload = asdict(capture_state.state) if capture_state is not None else {}
        statistics = dict(capture_payload.get("statistics", {}))
        pipeline = getattr(self, "pipeline", None)
        pipeline_payload = pipeline.status() if pipeline is not None else {}
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
            statistics["control_observe_fps"] = statistics["control_observation_fps"]
            statistics["queue_latency"] = getattr(pipeline_stats, "queue_latency_ms", 0.0)
            statistics["inference_latency"] = getattr(pipeline_stats, "inference_latency_ms", 0.0)
            statistics["inference_ms"] = getattr(pipeline_stats, "inference_latency_ms", 0.0)
            statistics["e2e_latency"] = getattr(pipeline_stats, "e2e_latency_ms", 0.0)
        else:
            statistics.setdefault("control_observe_fps", 0.0)
            statistics.setdefault("inference_ms", 0.0)
        deepstream_status = (
            pipeline_payload.get("deepstream", {}) if isinstance(pipeline_payload, dict) else {}
        )
        if isinstance(deepstream_status, dict) and deepstream_status:
            mailbox_status = deepstream_status.get("detection_batch_mailbox", {})
            mailbox_status = mailbox_status if isinstance(mailbox_status, dict) else {}
            stale_dropped = int(deepstream_status.get("stale_dropped_batches") or 0)
            timestamp_rejected = int(deepstream_status.get("timestamp_rejected_batches") or 0)
            non_monotonic_dropped = int(deepstream_status.get("non_monotonic_dropped_batches") or 0)
            mailbox_overwritten = int(mailbox_status.get("overwritten_batches") or 0)
            statistics["capture_counter"] = int(
                deepstream_status.get("capture_frames") or 0
            )
            statistics["capture_fps"] = float(
                deepstream_status.get("capture_fps") or 0.0
            )
            statistics["nvinfer_input_counter"] = int(
                deepstream_status.get("input_frames") or 0
            )
            statistics["nvinfer_input_fps"] = float(
                deepstream_status.get("input_fps") or 0.0
            )
            statistics["inference_counter"] = int(deepstream_status.get("output_buffers") or 0)
            statistics["detection_batch_counter"] = int(
                deepstream_status.get("published_batches") or 0
            )
            statistics["detection_batch_consumed_counter"] = int(
                getattr(pipeline_stats, "processed_frames", 0)
            )
            statistics["inference_fps"] = float(deepstream_status.get("output_fps") or 0.0)
            statistics["detection_batch_fps"] = float(deepstream_status.get("published_fps") or 0.0)
            statistics["control_observation_counter"] = int(
                getattr(pipeline_stats, "control_observations", 0)
            )
            statistics["stale_dropped_batches"] = stale_dropped
            statistics["timestamp_rejected_batches"] = timestamp_rejected
            statistics["non_monotonic_dropped_batches"] = non_monotonic_dropped
            statistics["mailbox_overwritten_batches"] = mailbox_overwritten
            statistics["skipped_counter"] = (
                stale_dropped + timestamp_rejected + non_monotonic_dropped + mailbox_overwritten
            )
            statistics["timestamp_source"] = str(deepstream_status.get("timestamp_source") or "")
            statistics["last_frame_age_ms"] = float(
                deepstream_status.get("latest_frame_age_ms") or 0.0
            )
            statistics["batch_age_ms"] = float(deepstream_status.get("last_batch_age_ms") or 0.0)
            nvinfer_ms = _status_statistic(
                deepstream_status,
                "nvinfer_stage_ms_stats",
                "p50",
            )
            batch_age_p50 = _status_statistic(
                deepstream_status,
                "batch_age_ms_stats",
                "p50",
            )
            build_ms = _status_statistic(
                deepstream_status,
                "detection_batch_build_ms_stats",
                "p50",
            )
            parser_status = deepstream_status.get("parser", {})
            parser_status = parser_status if isinstance(parser_status, dict) else {}
            parser_decode_ms = float(parser_status.get("decode_ms") or 0.0)
            statistics["inference_latency"] = nvinfer_ms
            statistics["inference_ms"] = nvinfer_ms
            statistics["e2e_latency"] = batch_age_p50
            statistics["stage_engine_ms"] = nvinfer_ms
            statistics["stage_decode_ms"] = parser_decode_ms
            statistics["stage_postprocess_ms"] = build_ms
        statistics["stale_drop_count"] = int(self.stale_drop_count)
        latest_frame_age_ms = self._latest_frame_age_ms()
        if latest_frame_age_ms is not None:
            statistics["latest_frame_age_ms"] = latest_frame_age_ms
        if inference_observation:
            if "frame_age_ms" in inference_observation:
                statistics["batch_age_ms"] = float(inference_observation.get("frame_age_ms") or 0.0)
            if "preprocess_ms" in inference_observation:
                statistics["preprocess_ms"] = float(
                    inference_observation.get("preprocess_ms") or 0.0
                )
            if "h2d_ms" in inference_observation:
                statistics["h2d_ms"] = float(inference_observation.get("h2d_ms") or 0.0)
            if (
                "frame_copy_cost_ms" in inference_observation
                or "frame_userspace_process_ms" in inference_observation
            ):
                statistics["host_frame_copy_ms"] = float(
                    inference_observation.get("frame_copy_cost_ms")
                    or inference_observation.get("frame_userspace_process_ms")
                    or 0.0
                )
        for key, value in pipeline_timings.items():
            statistics[f"stage_{key}"] = value
        statistics["postprocess_ms"] = float(pipeline_timings.get("postprocess_ms", 0.0))
        if capture_payload:
            capture_payload["statistics"] = statistics
        return RuntimeState(
            running=self.running,
            source=self.config.source.default,
            active_model=active_model,
            executor=self.executors.status(),
            capture=capture_payload,
            statistics=statistics,
            inference=self._runtime_inference_status(
                inference_state,
                active_model,
                pipeline_payload=pipeline_payload,
            ),
            config=self.config_store.status(),
            pipeline=pipeline_payload,
            vision=vision,
            fatal_error=self.fatal_error,
        )

    def _runtime_inference_status(
        self,
        inference_state: Any,
        active_model: dict | None,
        *,
        pipeline_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        backend = str(getattr(self.config.inference, "backend", "")).lower()
        status = inference_state.status() if inference_state is not None else {"available": False}
        payload = dict(status) if isinstance(status, dict) else {"available": False}
        if backend == "deepstream_nvinfer" and self.pipeline is not None:
            pipeline_status = (
                pipeline_payload
                if isinstance(pipeline_payload, dict)
                else self.pipeline.status()
            )
            deepstream_status = (
                pipeline_status.get("deepstream", {}) if isinstance(pipeline_status, dict) else {}
            )
            if isinstance(deepstream_status, dict) and deepstream_status:
                payload = {**payload, **deepstream_status}
                payload["loaded"] = bool(deepstream_status.get("running"))
                payload["available"] = bool(deepstream_status.get("available"))
        engine_selected = str(payload.get("selected") or "")
        payload["selected"] = backend
        payload["backend"] = backend
        if engine_selected and engine_selected != backend:
            payload["execution_backend"] = engine_selected
        payload["configured"] = active_model is not None
        payload.setdefault("running", bool(self.running))
        payload.setdefault("loaded", bool(payload.get("available", False)))
        payload["capture_memory"] = str(getattr(self.config.capture, "memory", ""))
        payload["capture_backend"] = str(getattr(self.config.capture, "backend", ""))
        payload["preprocess_backend"] = str(getattr(self.config.preprocess, "backend", ""))
        payload["device"] = str(getattr(self.config.inference, "device", "cuda"))
        payload["require_gpu"] = bool(getattr(self.config.inference, "require_gpu", True))
        payload["allow_cpu_fallback"] = bool(
            getattr(self.config.inference, "allow_cpu_fallback", False)
        )
        payload.setdefault(
            "reason",
            "DeepStream NVMM + nvinfer + C++ object-meta parser",
        )
        return {
            **payload,
        }

    def update_config(
        self,
        config: RuntimeConfig,
        *,
        executors: ExecutorRegistry | None = None,
    ) -> RuntimeConfig:
        mode_changed = str(config.control.mode) != str(self.config.control.mode)
        new_calibration_signature = self._config_calibration_signature(config)
        calibration_changed = new_calibration_signature != self._runtime_calibration_signature
        reset_reason = (
            "CONTROL_MODE_CHANGED"
            if mode_changed
            else "CALIBRATION_PROFILE_CHANGED"
            if calibration_changed
            else "CONTROL_CONFIG_UPDATED"
        )
        with self._control_lock:
            if executors is not None:
                self.executors = executors
            self.config = config
            self.control_timing.reset()
            self.last_control_timing = {}
            self.control_algorithms.reset()
            self.control_algorithms = self._create_control_algorithm_registry(config)
            self._clear_pending_commands(reset_reason)
            self.executors.update_runtime_config(config)
            if calibration_changed:
                self._runtime_calibration_signature = new_calibration_signature
                self._reset_detection_batch_cursor()
            self._reset_runtime_control_state(reset_reason)
        configure = getattr(self.inference, "configure", None)
        if callable(configure):
            configure(
                confidence_threshold=config.inference.confidence_threshold,
                nms_threshold=config.inference.nms_threshold,
                gpu_preprocessor=create_gpu_resource_preprocessor(config),
            )
        self._log_production_control_chain(
            "control_mode_changed"
            if mode_changed
            else "calibration_profile_changed"
            if calibration_changed
            else "config_updated"
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
    def _active_dual_phase_config(config: RuntimeConfig) -> Any:
        return config.control.dual_phase_atan_robust_predictive_v2

    def _is_robust_predictive_active(self) -> bool:
        return self.control_algorithms.is_active(DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2)

    def _active_mouse_controller(self) -> MouseController:
        controller = self.control_algorithms.active_controller
        if not isinstance(controller, MouseController):
            raise RuntimeError("active control algorithm is not a mouse controller")
        return controller

    def _active_robust_predictive_controller(
        self,
    ) -> DualPhaseAtanRobustPredictiveV2Algorithm:
        controller = self.control_algorithms.active_controller
        if not isinstance(controller, DualPhaseAtanRobustPredictiveV2Algorithm):
            raise RuntimeError("active control algorithm is not robust predictive v2")
        return controller

    @staticmethod
    def _config_calibration_signature(config: RuntimeConfig) -> tuple[Any, ...]:
        algorithm_id = str(config.control.active_algorithm)
        capture = config.capture
        roi = config.roi
        geometry_signature: tuple[Any, ...] = (
            algorithm_id,
            int(capture.width),
            int(capture.height),
            int(roi.size),
        )
        if algorithm_id == DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2:
            calibration = config.calibration
            projection = config.control.dual_phase_atan_robust_predictive_v2.projection
            return geometry_signature + (
                str(calibration.profile_id).strip(),
                int(calibration.profile_version),
                str(calibration.game_sensitivity_fingerprint).strip(),
                float(projection.fov_x_deg),
                float(projection.counts_per_360),
                bool(projection.invert_y),
            )
        if algorithm_id == CALIBRATED_ANGULAR:
            calibration = config.calibration
            calibrated = config.control.calibrated_angular
            return geometry_signature + (
                str(calibration.profile_id).strip(),
                int(calibration.profile_version),
                str(calibration.game_sensitivity_fingerprint).strip(),
                float(calibrated.fov_x_deg),
                float(calibrated.counts_per_360_x),
                float(calibrated.counts_per_360_y),
                bool(config.control.shared.invert_y),
            )
        return geometry_signature

    def _reset_runtime_control_state(self, reason: str) -> None:
        self._last_runtime_reset_reason = reason
        self.target_selector.reset()
        self.control_timing.reset()
        self.last_control_timing = {}
        self._reset_control_motion_state()
        self._reset_direct_command_executor()
        self._clear_pending_commands(reason)
        self.last_frame_context = None
        self.last_target = None
        self.last_execution = None
        self._last_control_tick_ns = 0
        self._executed_control_samples.clear()
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
        if reason in {"RUNTIME_STOPPED", "RUNTIME_FATAL_ERROR"}:
            self.last_inference_reason = reason
            if isinstance(self.last_inference_status, dict):
                self.last_inference_status.update(
                    {
                        "available": False,
                        "reason": reason,
                        "terminal_rejected": True,
                    }
                )

    def _log_production_control_chain(self, event: str) -> None:
        mode = self.config.control.active_algorithm
        selected_executor = getattr(self.executors, "selected", None) or "kmnet"
        is_dual_phase = self._is_robust_predictive_active()
        if is_dual_phase:
            projection = self._active_dual_phase_config(self.config).projection
            fov_x_deg = projection.fov_x_deg
            counts_per_360_x = projection.counts_per_360
            counts_per_360_y = projection.counts_per_360
            invert_y = projection.invert_y
        elif mode == CALIBRATED_ANGULAR:
            calibrated = self.config.control.calibrated_angular
            fov_x_deg = calibrated.fov_x_deg
            counts_per_360_x = calibrated.counts_per_360_x
            counts_per_360_y = calibrated.counts_per_360_y
            invert_y = self.config.control.shared.invert_y
        else:
            fov_x_deg = 0.0
            counts_per_360_x = 0.0
            counts_per_360_y = 0.0
            invert_y = self.config.control.shared.invert_y
        delivery_stage = (
            "LatestReplaceScheduler"
            if is_dual_phase
            else "CommandScheduler"
            if bool(self.config.control.scheduler_enabled)
            else "DirectObservationSend"
        )
        if mode == DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2:
            chain = (
                "MeasuredAim->FAR/NEAR->FourPointMedianVelocity->AdaptiveEMA"
                "->BoundedXPrediction->ProjectionAngle->FullCounts->CountsAtan"
                f"->IntegerQuantizer->{delivery_stage}->kmNet"
            )
        elif is_dual_phase:
            chain = (
                "RawAimObservation->RealError+BoundedXPrediction->FAR/NEARCountsAtan"
                f"->IntegerQuantizer->{delivery_stage}->kmNet"
            )
        else:
            chain = (
                "RawAimPoint->MeasuredPixelError->ExclusiveController"
                f"->SharedCountLimits->{delivery_stage}->kmNet"
            )
        logger.info(
            "production_control_chain event=%s chain=%s controller=%s executor=%s trigger=%s "
            "calibration_profile_id=%s calibration_profile_version=%s fov_x_deg=%.3f "
            "counts_per_360_x=%.3f counts_per_360_y=%.3f invert_y=%s",
            event,
            chain,
            mode,
            selected_executor,
            self.config.control.trigger_mode,
            self.config.calibration.profile_id,
            self.config.calibration.profile_version,
            float(fov_x_deg),
            float(counts_per_360_x),
            float(counts_per_360_y),
            bool(invert_y),
        )

    def _calibration_fingerprint_status(self) -> dict[str, Any]:
        expected = str(self.config.calibration.game_sensitivity_fingerprint).strip()
        observed = str(self._external_sensitivity_fingerprint).strip()
        source = str(self._external_sensitivity_source).strip()
        updated_ts_ns = int(self._external_sensitivity_ts_ns or 0)
        if not self.control_algorithms.active_definition.capabilities.requires_calibration:
            return {
                "state": "not_required",
                "control_allowed": True,
                "expected_fingerprint": expected,
                "observed_fingerprint": observed,
                "source": source,
                "updated_ts_ns": updated_ts_ns,
                "reason_code": "",
                "reason": f"{self.config.control.active_algorithm} does not require angular calibration",
            }
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
        with self._control_lock:
            self.running = False
            self.fatal_error = {
                "type": "FATAL_ERROR",
                "thread": thread_name,
                "message": str(exc),
                "crash_log": str(path),
            }
            self._reset_detection_batch_cursor()
            self._reset_runtime_control_state("RUNTIME_FATAL_ERROR")

    def process_frame(self, context: FrameContext) -> RuntimeFrameResult:
        execution_payload: dict[str, Any] | None = None
        with self._control_lock:
            if not self.running or self.fatal_error is not None:
                reason = (
                    "RUNTIME_FATAL_ERROR" if self.fatal_error is not None else "RUNTIME_STOPPED"
                )
                self._reset_detection_batch_cursor()
                self._reset_runtime_control_state(reason)
                return self._empty_runtime_frame_result()
            self.last_frame_context = context
            intent = self._control_intent_from_context(context)
            self._last_control_tick_ns = time.monotonic_ns()
            control_intents = [intent] if intent is not None else []
            execution_results = [self.executors.execute(intent) for intent in control_intents]
            for result in execution_results:
                if self._is_robust_predictive_active() and not bool(
                    getattr(result, "sent", False)
                ) and not self._execution_is_pending_latest_replace(result):
                    # A failed or blocked device call must not leave fractional
                    # demand from an unsent observation to a later frame.
                    self._active_robust_predictive_controller().release_trigger()
                self._record_executed_control(result)
            if execution_results:
                self.last_execution = self._execution_result_payload(execution_results[-1])
                self._attach_execution_to_last_control(self.last_execution)
                execution_payload = dict(self.last_execution)
            elif isinstance(self.last_control, dict):
                if isinstance(self.last_execution, dict):
                    self._attach_execution_to_last_control(self.last_execution)
                else:
                    self.last_control.setdefault("execution", None)
                    self.last_control.setdefault("driver_counts", None)
                    self.last_control.setdefault("driver_dx", None)
                    self.last_control.setdefault("driver_dy", None)
            self._record_control_frame()
        if execution_results:
            assert execution_payload is not None
            logger.debug(
                "control execution result frame=%s executor=%s sent=%s dx=%.1f dy=%.1f message=%s meta=%s",
                context.frame_id,
                execution_payload.get("executor_id"),
                execution_payload.get("sent"),
                float(execution_payload.get("output_dx") or 0.0),
                float(execution_payload.get("output_dy") or 0.0),
                execution_payload.get("message"),
                execution_payload.get("metadata"),
            )
        return RuntimeFrameResult(
            control_intents=control_intents,
            execution_results=execution_results,
        )

    def process_control_tick(self) -> RuntimeFrameResult:
        if self._is_robust_predictive_active():
            with self._control_lock:
                if not self.running:
                    self._reset_control_motion_state()
                    self._reset_direct_command_executor()
                    return self._empty_runtime_frame_result()
                if (
                    str(self.config.control.trigger_mode) == "hardware"
                    and not self._box_input_state().active
                ):
                    self._active_robust_predictive_controller().release_trigger()
                    self._clear_pending_commands("TRIGGER_INACTIVE")
                    return self._empty_runtime_frame_result()
        tick_pending = getattr(self.executors, "tick_pending", None)
        if not callable(tick_pending):
            return self._empty_runtime_frame_result()
        with self._control_lock:
            if not self.running:
                self._reset_control_motion_state()
                self._clear_pending_commands("RUNTIME_STOPPED")
                return self._empty_runtime_frame_result()
            if (
                str(self.config.control.trigger_mode) == "hardware"
                and not self._box_input_state().active
            ):
                self._reset_control_motion_state()
                self._clear_pending_commands("TRIGGER_INACTIVE")
                return self._empty_runtime_frame_result()
        result = tick_pending()
        with self._control_lock:
            self._last_control_tick_ns = time.monotonic_ns()
            self._record_executed_control(result)
            if str(result.message) == "no pending control command ready":
                return self._empty_runtime_frame_result()
            self.last_execution = self._execution_result_payload(result)
            self._attach_execution_to_last_control(self.last_execution)
            self._record_control_frame()
        return RuntimeFrameResult(
            control_intents=[],
            execution_results=[result],
            observation_updated=False,
        )

    @staticmethod
    def _execution_is_pending_latest_replace(result: Any) -> bool:
        metadata = getattr(result, "metadata", {}) or {}
        return bool(
            isinstance(metadata, dict)
            and metadata.get("action") in {"replace_plan", "hold_latest"}
        )

    @staticmethod
    def _empty_runtime_frame_result() -> RuntimeFrameResult:
        return RuntimeFrameResult(
            control_intents=[], execution_results=[], observation_updated=False
        )

    def _finalize_detection_batch_early_result(self) -> RuntimeFrameResult:
        """Keep terminal runtime state authoritative over concurrent early exits."""

        with self._control_lock:
            terminal_reason = (
                "RUNTIME_FATAL_ERROR"
                if self.fatal_error is not None
                else "RUNTIME_STOPPED"
                if not self.running
                else ""
            )
            if terminal_reason:
                self._reset_detection_batch_cursor()
                self._reset_runtime_control_state(terminal_reason)
        return self._empty_runtime_frame_result()

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
        with self._control_lock:
            terminal_reason = (
                "RUNTIME_FATAL_ERROR"
                if self.fatal_error is not None
                else "RUNTIME_STOPPED"
                if not self.running
                else ""
            )
            if terminal_reason:
                done_ns = time.monotonic_ns()
                self._reset_detection_batch_cursor()
                self._reset_runtime_control_state(terminal_reason)
                self.last_inference_reason = terminal_reason
                self._set_detection_batch_pipeline_timings(
                    detection_batch,
                    total_start_ns=total_start_ns,
                    control_start_ns=None,
                    done_ns=done_ns,
                )
                self.last_inference_status = self._detection_batch_status_payload(
                    detection_batch,
                    width=int(width),
                    height=int(height),
                    now_ns=done_ns,
                    reason=terminal_reason,
                    available=False,
                    mapped_detections=0,
                    source_width=resolved_source_width,
                    source_height=resolved_source_height,
                    source_geometry_source=source_geometry_source,
                    source_geometry_trusted=source_geometry_trusted,
                    roi_offset_x=resolved_roi_offset_x,
                    roi_offset_y=resolved_roi_offset_y,
                    extra={"terminal_rejected": True},
                )
                return self._empty_runtime_frame_result()
        freshness_reason = self._detection_batch_freshness_reason(detection_batch)
        if freshness_reason:
            self.stale_drop_count += 1
            self._reset_runtime_control_state("DETECTION_BATCH_NOT_LATEST")
            self.last_inference_reason = freshness_reason
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
                extra={"latest_rejected": True},
            )
            return self._finalize_detection_batch_early_result()
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
            return self._finalize_detection_batch_early_result()
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
            return self._finalize_detection_batch_early_result()
        stale_reason = self._detection_batch_stale_reason(
            detection_batch,
            now_ns=total_start_ns,
        )
        if stale_reason:
            self.stale_drop_count += 1
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
            return self._finalize_detection_batch_early_result()
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
        )
        context = detection_batch_to_frame_context(
            detection_batch,
            width=int(width),
            height=int(height),
        )
        self._record_accepted_detection_batch(detection_batch)
        control_start_ns = time.monotonic_ns()
        result = self.process_frame(context)
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
        metadata = dict(getattr(detection_batch, "metadata", {}) or {})
        parser = metadata.get("parser")
        parser_payload = parser if isinstance(parser, dict) else {}
        decode_ms = float(parser_payload.get("decode_ms") or 0.0)
        nvinfer_stage_ms = float(
            metadata.get("nvinfer_stage_ms")
            or metadata.get("nvinfer_total_ms")
            or detection_batch.inference_latency_ms
        )
        self.last_pipeline_timings = {
            "roi_ms": 0.0,
            "engine_ms": nvinfer_stage_ms,
            # DeepStream exposes nvinfer sink-to-src elapsed time. It does not
            # expose isolated TensorRT execution time on this path.
            "engine_execute_ms": None,
            "decode_ms": decode_ms,
            "nms_ms": None,
            "detection_batch_build_ms": float(metadata.get("detection_batch_build_ms") or 0.0),
            "handoff_ms": max(
                0.0,
                (handoff_start_ns - int(detection_batch.inference_end_ts_ns)) / 1e6,
            ),
            "postprocess_ms": decode_ms,
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
        frame_age_ms = self._detection_batch_age_ms(detection_batch, now_ns=now_ns)
        configured_actuation_delay_s = max(
            0.0,
            float(getattr(self.config.control, "configured_actuation_delay_s", 0.0)),
        )
        payload: dict[str, object] = {
            "frame_id": detection_batch.frame_id,
            "generation": int(detection_batch.generation or 0),
            "source_sequence": int(detection_batch.source_sequence or 0),
            "capture_ts_ns": detection_batch.capture_ts_ns,
            "publish_ts_ns": int(detection_batch.publish_ts_ns),
            "clock_domain": str(detection_batch.clock_domain),
            "input_age_ms": float(detection_batch.input_age_ms),
            "inference_ms": float(
                detection_batch.inference_ms or detection_batch.inference_latency_ms
            ),
            "result_age_ms": float(detection_batch.result_age_ms),
            "is_stale": bool(detection_batch.is_stale),
            "control_now_ts_ns": int(now_ns),
            "target_id": None,
            "measurement_dt_ms": None,
            "frame_age_ms": frame_age_ms,
            "configured_actuation_delay_s": configured_actuation_delay_s,
            "actuation_delay_source": "configured_estimate",
            "prediction_horizon_ms": frame_age_ms + configured_actuation_delay_s * 1000.0,
            "ran": True,
            "available": bool(available),
            "reason": str(reason),
            "detection_coordinate_space": detection_batch.coordinate_space,
            "raw_detections": len(detection_batch.detections),
            "mapped_detections": int(mapped_detections),
            "detection_batch_frame_id": detection_batch.frame_id,
            "detection_batch_generation": int(detection_batch.generation or 0),
            "detection_batch_capture_ts_ns": detection_batch.capture_ts_ns,
            "inference_start_ts_ns": detection_batch.inference_start_ts_ns,
            "inference_end_ts_ns": detection_batch.inference_end_ts_ns,
            "detection_batch_inference_latency_ms": detection_batch.inference_latency_ms,
            "detection_batch_model_input_size": (
                list(detection_batch.model_input_size)
                if detection_batch.model_input_size is not None
                else []
            ),
            "detection_batch_roi_size": (
                list(detection_batch.roi_size)
                if detection_batch.roi_size is not None
                else [int(width), int(height)]
            ),
            "detection_batch_metadata": dict(getattr(detection_batch, "metadata", {}) or {}),
            "latency_source": (
                "capture_to_object_meta_done"
                if str((getattr(detection_batch, "metadata", {}) or {}).get("source", ""))
                == "deepstream_nvinfer"
                else "custom_tensorrt_done"
            ),
            "classes": list(detection_batch.classes),
            "input_width": int(width),
            "input_height": int(height),
            "model_input_width": int(
                detection_batch.model_input_size[0]
                if detection_batch.model_input_size is not None
                else width
            ),
            "model_input_height": int(
                detection_batch.model_input_size[1]
                if detection_batch.model_input_size is not None
                else height
            ),
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
        self._log_control_timing_payload(payload, event="detection_batch")
        return payload

    def _detection_batch_freshness_reason(self, detection_batch: DetectionBatch) -> str:
        if bool(getattr(detection_batch, "is_stale", False)):
            return "DetectionBatch marked stale by producer"
        generation = int(getattr(detection_batch, "generation", detection_batch.frame_id) or 0)
        if generation <= self._accepted_batch_generation:
            return (
                "DetectionBatch generation must increase: "
                f"{generation} <= {self._accepted_batch_generation}"
            )
        capture_ts_ns = int(detection_batch.capture_ts_ns)
        if capture_ts_ns <= self._accepted_batch_capture_ts_ns:
            return (
                "DetectionBatch capture_ts_ns must increase: "
                f"{capture_ts_ns} <= {self._accepted_batch_capture_ts_ns}"
            )
        return ""

    def _detection_batch_latest_generation_reason(
        self,
        detection_batch: DetectionBatch,
        *,
        latest_generation: int | None,
        latest_frame_id: int | None = None,
    ) -> str:
        if latest_generation is None:
            generation_reason = ""
        else:
            generation = int(getattr(detection_batch, "generation", detection_batch.frame_id) or 0)
            if generation < int(latest_generation):
                generation_reason = (
                    "DetectionBatch generation is no longer latest generation: "
                    f"{generation} < {int(latest_generation)}"
                )
            else:
                generation_reason = ""
        if generation_reason:
            return generation_reason
        if latest_frame_id is None:
            return ""
        frame_id = int(getattr(detection_batch, "frame_id", 0) or 0)
        if frame_id < int(latest_frame_id):
            return (
                "DetectionBatch frame_id is no longer latest frame_id: "
                f"{frame_id} < {int(latest_frame_id)}"
            )
        return ""

    def _latest_published_identity(self) -> tuple[int | None, int | None]:
        capture = getattr(self, "capture", None)
        broker = getattr(capture, "latest_frame_broker", None)
        if broker is None:
            session = getattr(capture, "session", None)
            broker = getattr(session, "latest_frame_broker", None)
        status_fn = getattr(broker, "status", None)
        if not callable(status_fn):
            return None, None
        try:
            status = status_fn()
        except Exception:
            return None, None
        if not isinstance(status, dict):
            return None, None
        return (
            self._latest_status_int(status, ("published_generation", "latest_generation")),
            self._latest_status_int(status, ("published_frame_id", "latest_frame_id")),
        )

    def _latest_status_int(
        self,
        status: dict[str, Any],
        keys: tuple[str, ...],
    ) -> int | None:
        for key in keys:
            value = status.get(key)
            if value is None:
                continue
            try:
                parsed = int(value)
            except Exception:
                continue
            return parsed if parsed >= 0 else None
        return None

    def _latest_published_generation(self) -> int | None:
        latest_generation, _latest_frame_id = self._latest_published_identity()
        return latest_generation

    def _latest_published_frame_id(self) -> int | None:
        _latest_generation, latest_frame_id = self._latest_published_identity()
        return latest_frame_id

    def _latest_frame_age_ms(self) -> float | None:
        capture = getattr(self, "capture", None)
        latest_frame = None
        latest_frame_fn = getattr(capture, "get_latest_preview_frame", None)
        if callable(latest_frame_fn):
            try:
                latest_frame = latest_frame_fn()
            except Exception:
                latest_frame = None
        if latest_frame is None:
            session = getattr(capture, "session", None)
            latest_frame_fn = getattr(session, "latest_frame", None)
            if callable(latest_frame_fn):
                try:
                    latest_frame = latest_frame_fn(timeout_s=0.0)
                except Exception:
                    latest_frame = None
        if latest_frame is None:
            return None
        try:
            capture_ts_ns = int(getattr(latest_frame, "capture_ts_ns"))
        except Exception:
            return None
        return max(0.0, (time.monotonic_ns() - capture_ts_ns) / 1e6)

    @staticmethod
    def _model_input_size_from_debug(
        debug: dict[str, Any] | None,
        *,
        fallback_width: int,
        fallback_height: int,
    ) -> tuple[int, int]:
        debug_payload = debug if isinstance(debug, dict) else {}
        preprocess = debug_payload.get("preprocess")
        preprocess_debug = preprocess if isinstance(preprocess, dict) else {}
        size = preprocess_debug.get("model_input_size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            try:
                width = int(size[0])
                height = int(size[1])
                if width > 0 and height > 0:
                    return width, height
            except Exception:
                pass
        try:
            width = int(preprocess_debug.get("model_width") or fallback_width)
            height = int(preprocess_debug.get("model_height") or fallback_height)
        except Exception:
            width = int(fallback_width)
            height = int(fallback_height)
        return max(1, width), max(1, height)

    def _record_accepted_detection_batch(self, detection_batch: DetectionBatch) -> None:
        self._accepted_batch_generation = int(
            getattr(detection_batch, "generation", detection_batch.frame_id) or 0
        )
        self._accepted_batch_capture_ts_ns = int(detection_batch.capture_ts_ns)

    def _reset_detection_batch_cursor(self) -> None:
        self._accepted_batch_generation = -1
        self._accepted_batch_capture_ts_ns = 0

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
                    "detection_batch",
                )
        capture_config = getattr(self.config, "capture", None)
        config_source_width = int(getattr(capture_config, "width", 0) or 0)
        config_source_height = int(getattr(capture_config, "height", 0) or 0)
        if config_source_width <= 0 or config_source_height <= 0:
            return 0, 0, 0, 0, False, "missing_source_geometry"
        requested_size = max(1, min(int(width), int(height)))
        roi_x, roi_y, _roi_size = center_roi_region(
            source_width=config_source_width,
            source_height=config_source_height,
            requested_size=requested_size,
        )
        return (
            config_source_width,
            config_source_height,
            int(roi_x),
            int(roi_y),
            True,
            "runtime_config",
        )

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

    def _detection_batch_stale_reason(
        self,
        detection_batch: DetectionBatch,
        *,
        now_ns: int,
    ) -> str:
        gate = FreshnessGate.strictest(
            getattr(getattr(self.config, "runtime", None), "freshness_threshold_ms", 0.0),
        )
        return gate.stale_reason(
            capture_ts_ns=getattr(detection_batch, "capture_ts_ns", 0),
            now_ns=now_ns,
            template="DetectionBatch frame age exceeds control latency guard: {age_ms:.1f}ms > {threshold_ms:.1f}ms",
        )

    @staticmethod
    def _detection_batch_age_ms(
        detection_batch: DetectionBatch,
        *,
        now_ns: int,
    ) -> float:
        return FreshnessGate.age_ms(
            capture_ts_ns=getattr(detection_batch, "capture_ts_ns", 0),
            now_ns=now_ns,
        )

    def _record_control_timing(
        self,
        context: FrameContext,
        *,
        target: Track | None,
        control_now_ts_ns: int,
    ) -> dict[str, Any]:
        track_id = target.track_id if target is not None else None
        snapshot = self.control_timing.observe(
            frame_id=context.frame_id,
            target_id=int(track_id) if track_id is not None else None,
            capture_ts_ns=int(context.capture_ts_ns or 0),
            inference_end_ts_ns=context.inference_end_ts_ns,
            control_now_ts_ns=control_now_ts_ns,
            configured_actuation_delay_s=float(self.config.control.configured_actuation_delay_s),
        )
        payload = snapshot.as_telemetry()
        self.last_control_timing = payload
        if isinstance(self.last_inference_status, dict):
            self.last_inference_status.update(payload)
        self._log_control_timing_payload(payload, event="target_observation")
        return payload

    @staticmethod
    def _log_control_timing_payload(payload: dict[str, Any], *, event: str) -> None:
        logger.debug(
            "control_timing event=%s frame=%s target=%s capture_ts_ns=%s "
            "inference_end_ts_ns=%s "
            "control_now_ts_ns=%s measurement_dt_ms=%s frame_age_ms=%.3f "
            "configured_actuation_delay_s=%.6f prediction_horizon_ms=%.3f",
            event,
            payload["frame_id"],
            payload["target_id"],
            payload["capture_ts_ns"],
            payload["inference_end_ts_ns"],
            payload["control_now_ts_ns"],
            payload["measurement_dt_ms"],
            float(payload["frame_age_ms"] or 0.0),
            float(payload["configured_actuation_delay_s"] or 0.0),
            float(payload["prediction_horizon_ms"] or 0.0),
        )

    def process_captured_frame(
        self,
        frame: CapturedFrame,
        *,
        acquired_generation: int | None = None,
    ) -> RuntimeFrameResult:
        total_start_ns = time.monotonic_ns()
        if self.inference is None:
            self._record_inference_status(
                frame=frame,
                ran=False,
                available=False,
                reason="推理运行时未初始化",
            )
            result = self.process_frame(self._empty_frame_context(frame))
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
            result = self.process_frame(self._empty_frame_context(frame))
            self._record_pipeline_timings(total_start_ns, control_start_ns=total_start_ns)
            return result

        try:
            roi_start_ns = time.monotonic_ns()
            roi_frame = center_roi_frame(
                frame,
                requested_size=self.config.roi.size,
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
            result = self.process_frame(self._empty_frame_context(frame))
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
            result = self.process_frame(self._empty_frame_context(frame))
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
            result = self.process_frame(self._empty_frame_context(frame))
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
            model_input_size = self._model_input_size_from_debug(
                inference_result.debug,
                fallback_width=int(getattr(roi_frame, "width", frame.width)),
                fallback_height=int(getattr(roi_frame, "height", frame.height)),
            )
            generation = int(getattr(frame, "generation", 0) or frame.frame_id)
            detection_batch = DetectionBatch(
                frame_id=frame.frame_id,
                capture_ts_ns=frame.capture_ts_ns,
                inference_start_ts_ns=infer_start_ns,
                inference_end_ts_ns=postprocess_start_ns,
                detections=detections,
                classes=classes,
                coordinate_space="roi",
                generation=generation,
                publish_ts_ns=postprocess_start_ns,
                input_age_ms=max(0.0, (infer_start_ns - int(frame.capture_ts_ns)) / 1e6),
                inference_ms=max(0.0, (postprocess_start_ns - infer_start_ns) / 1e6),
                result_age_ms=max(0.0, (postprocess_start_ns - int(frame.capture_ts_ns)) / 1e6),
                source_sequence=generation,
                is_stale=False,
                clock_domain="monotonic",
                model_input_size=model_input_size,
                metadata={
                    "source_backend": str(getattr(frame, "source_backend", "") or ""),
                    "caps_string": str(getattr(frame, "caps_string", "") or ""),
                    "model_input_size": model_input_size,
                },
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
            result = self.process_frame(self._empty_frame_context(frame))
            self._record_pipeline_timings(
                total_start_ns,
                roi_start_ns=roi_start_ns,
                infer_start_ns=infer_start_ns,
                postprocess_start_ns=postprocess_start_ns,
                control_start_ns=control_start_ns,
            )
            return result
        acquired_generation = (
            int(acquired_generation)
            if acquired_generation is not None
            else int(getattr(frame, "generation", 0) or frame.frame_id)
        )
        acquired_frame_id = int(getattr(frame, "frame_id", 0) or acquired_generation)
        broker_published_generation, broker_published_frame_id = self._latest_published_identity()
        published_generation = (
            broker_published_generation
            if broker_published_generation is not None
            else acquired_generation
        )
        published_frame_id = (
            broker_published_frame_id
            if broker_published_frame_id is not None
            else acquired_frame_id
        )
        detection_generation = int(
            detection_batch.frame_id
            if detection_batch.generation is None
            else detection_batch.generation
        )
        freshness_extra = {
            "acquired_generation": acquired_generation,
            "acquired_frame_id": acquired_frame_id,
            "latest_generation": acquired_generation,
            "latest_frame_id": acquired_frame_id,
            "broker_published_generation": published_generation,
            "broker_published_frame_id": published_frame_id,
            "generation_lag": max(0, int(published_generation) - detection_generation),
            "frame_id_lag": max(0, int(published_frame_id) - int(detection_batch.frame_id)),
            "published_since_acquire": max(0, int(published_generation) - acquired_generation),
        }
        freshness_reason = self._detection_batch_freshness_reason(detection_batch)
        if not freshness_reason:
            freshness_reason = self._detection_batch_latest_generation_reason(
                detection_batch,
                latest_generation=acquired_generation,
                latest_frame_id=acquired_frame_id,
            )
        if not freshness_reason:
            freshness_reason = self._detection_batch_stale_reason(
                detection_batch,
                now_ns=postprocess_start_ns,
            )
        if freshness_reason:
            self.stale_drop_count += 1
            self._reset_runtime_control_state("DETECTION_BATCH_NOT_LATEST")
            self.last_inference_reason = freshness_reason
            self._record_inference_status(
                frame=frame,
                roi_frame=roi_frame,
                ran=True,
                available=False,
                reason=freshness_reason,
                raw_detections=len(inference_result.detections),
                mapped_detections=0,
                classes=detection_batch.classes,
                detection_batch=detection_batch,
                debug=inference_result.debug,
                extra={
                    "stale_rejected": True,
                    "stale_drop_count": self.stale_drop_count,
                    **freshness_extra,
                },
            )
            control_start_ns = time.monotonic_ns()
            result = self._empty_runtime_frame_result()
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
            extra=freshness_extra,
        )

        context = detection_batch_to_frame_context(
            detection_batch,
            width=roi_frame.width,
            height=roi_frame.height,
        )
        self._record_accepted_detection_batch(detection_batch)
        control_start_ns = time.monotonic_ns()
        result = self.process_frame(context)
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
        debug_timings = (
            inference_debug.get("timings", {}) if isinstance(inference_debug, dict) else {}
        )
        decode_debug = (
            inference_debug.get("decode", {}) if isinstance(inference_debug, dict) else {}
        )
        decode_timings = decode_debug.get("timings", {}) if isinstance(decode_debug, dict) else {}
        decode_ms = float(decode_timings.get("decode_ms") or debug_timings.get("decode_ms") or 0.0)
        preprocess_ms = float(
            debug_timings.get("native_preprocess_total_ms")
            or debug_timings.get("numpy_tensor_ms")
            or 0.0
        )
        h2d_ms = float(decode_timings.get("h2d_enqueue_ms") or 0.0)
        engine_total_ms = float(
            decode_timings.get("total_ms") or debug_timings.get("execute_total_ms") or 0.0
        )
        engine_execute_ms = max(0.0, engine_total_ms - decode_ms)
        self.last_pipeline_timings = {
            "roi_ms": ms(roi_start_ns, infer_start_ns),
            "preprocess_ms": preprocess_ms,
            "h2d_ms": h2d_ms,
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
            generation=int(getattr(frame, "generation", 0) or frame.frame_id),
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
        extra: dict[str, object] | None = None,
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
        decode_debug = debug_payload.get("decode")
        decode_timings = decode_debug.get("timings", {}) if isinstance(decode_debug, dict) else {}
        debug_timings = (
            debug_payload.get("timings", {})
            if isinstance(debug_payload.get("timings"), dict)
            else {}
        )
        preprocess_ms = float(
            debug_timings.get("native_preprocess_total_ms")
            or debug_timings.get("numpy_tensor_ms")
            or 0.0
        )
        h2d_ms = float(decode_timings.get("h2d_enqueue_ms") or 0.0)
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
            "frame_copy_cost_ms": float(
                getattr(frame, "copy_cost_ms", getattr(frame, "userspace_process_ms", 0.0)) or 0.0
            ),
            "frame_resource_kind": str(getattr(frame_resource, "kind", "") or ""),
            "frame_resource_memory": str(getattr(frame_resource, "memory", "") or "cpu"),
            "frame_dmabuf_fd": getattr(frame_resource, "dmabuf_fd", None),
            "frame_age_ms": max(0.0, (time.monotonic_ns() - int(frame.capture_ts_ns)) / 1e6),
            "preprocess_ms": preprocess_ms,
            "h2d_ms": h2d_ms,
            "ran": ran,
            "available": available,
            "reason": reason,
            "raw_detections": raw_detections,
            "mapped_detections": mapped_detections,
            "detection_batch_frame_id": detection_batch.frame_id
            if detection_batch is not None
            else 0,
            "detection_batch_generation": int(detection_batch.generation or 0)
            if detection_batch is not None
            else 0,
            "detection_batch_capture_ts_ns": detection_batch.capture_ts_ns
            if detection_batch is not None
            else 0,
            "detection_batch_publish_ts_ns": detection_batch.publish_ts_ns
            if detection_batch is not None
            else 0,
            "detection_batch_source_sequence": int(detection_batch.source_sequence or 0)
            if detection_batch is not None
            else 0,
            "detection_batch_clock_domain": detection_batch.clock_domain
            if detection_batch is not None
            else "",
            "detection_batch_input_age_ms": detection_batch.input_age_ms
            if detection_batch is not None
            else 0.0,
            "detection_batch_result_age_ms": detection_batch.result_age_ms
            if detection_batch is not None
            else 0.0,
            "detection_batch_is_stale": detection_batch.is_stale
            if detection_batch is not None
            else False,
            "detection_batch_model_input_size": (
                list(detection_batch.model_input_size)
                if detection_batch is not None and detection_batch.model_input_size is not None
                else []
            ),
            "inference_start_ts_ns": detection_batch.inference_start_ts_ns
            if detection_batch is not None
            else 0,
            "inference_end_ts_ns": detection_batch.inference_end_ts_ns
            if detection_batch is not None
            else 0,
            "detection_batch_inference_latency_ms": detection_batch.inference_latency_ms
            if detection_batch is not None
            else 0.0,
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
            "detection_coordinate_space": detection_batch.coordinate_space
            if detection_batch is not None
            else "roi",
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
            "debug": debug_payload,
        }
        if extra:
            self.last_inference_status.update(extra)

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
            self._reset_control_motion_state()
            self._log_no_control_target(context, selection)
            self.last_target = None
            selector_debug = dict(getattr(self.target_selector, "last_debug", {}) or {})
            track_diagnostics = self._track_diagnostics_payload(
                selector_debug,
                selection=selection,
            )
            control_now_ns = time.monotonic_ns()
            self.last_control = {
                "frame_id": context.frame_id,
                "control_now_ts_ns": control_now_ns,
                "trajectory_generation": int(
                    context.frame_id if context.generation is None else context.generation
                ),
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

        trigger_mode = str(getattr(self.config.control, "trigger_mode", "always") or "always")
        if trigger_mode not in {"hardware", "always"}:
            trigger_mode = "hardware"
        shared_control = self.config.control.shared
        hardware_input = (
            self._box_input_state()
            if trigger_mode == "hardware"
            or bool(shared_control.recoil_enabled)
            else BoxInputState(raw={"source": "monitor_not_required"})
        )
        box_input = (
            BoxInputState(
                left=True,
                raw={
                    "source": "target_detection",
                    "active": True,
                    "reason": "target-driven control",
                },
            )
            if trigger_mode == "always"
            else hardware_input
        )
        target_key = self._control_target_key(target)
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
                "class_score": selection.class_score,
                "distance_score": selection.distance_score,
                "selection_score": selection.selection_score,
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
                "control_now_ts_ns": time.monotonic_ns(),
                "trajectory_generation": int(
                    context.frame_id if context.generation is None else context.generation
                ),
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
                "class_score": selection.class_score,
                "distance_score": selection.distance_score,
                "selection_score": selection.selection_score,
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
        control_metadata = self._control_frame_metadata(context)
        selector_debug = dict(getattr(self.target_selector, "last_debug", {}) or {})
        track_diagnostics = self._track_diagnostics_payload(
            selector_debug,
            target=target,
            selection=selection,
        )
        control_now_ns = time.monotonic_ns()
        trigger_hold_ms = self._box_input_hold_ms(box_input, control_now_ns)
        left_trigger_hold_ms = self._box_input_hold_ms(
            hardware_input,
            control_now_ns,
            left_only=True,
        )
        activation_delay_ms = max(
            0.0,
            float(shared_control.trigger_activation_delay_ms),
        )
        trigger_activation_ready = bool(
            trigger_mode == "always"
            or (box_input.active and trigger_hold_ms >= activation_delay_ms)
        )
        timing_payload = self._record_control_timing(
            context,
            target=target,
            control_now_ts_ns=control_now_ns,
        )
        measurement_dt_ms = timing_payload.get("measurement_dt_ms")
        algorithm_trigger_active = bool(
            trigger_activation_ready and (box_input.active or trigger_mode == "always")
        )
        if self._is_robust_predictive_active():
            command, observation_metadata = self._dual_phase_control_command(
                context=context,
                target=target,
                control_metadata=control_metadata,
                selector_debug=selector_debug,
                control_now_ts_ns=control_now_ns,
                trigger_active=algorithm_trigger_active,
                left_trigger_active=bool(hardware_input.left),
                left_trigger_hold_ms=left_trigger_hold_ms,
                measurement_dt_ms=(
                    float(measurement_dt_ms)
                    if isinstance(measurement_dt_ms, (int, float))
                    else None
                ),
            )
            control_metadata.update(observation_metadata)
        else:
            control_metadata.update(
                self._mouse_observation_metadata(
                    context=context,
                    target=target,
                    control_metadata=control_metadata,
                    selector_debug=selector_debug,
                    control_now_ts_ns=control_now_ns,
                    measurement_dt_s=(
                        float(measurement_dt_ms) / 1000.0
                        if isinstance(measurement_dt_ms, (int, float))
                        else None
                    ),
                    left_trigger_active=bool(hardware_input.left),
                    left_trigger_hold_ms=left_trigger_hold_ms,
                )
            )
            observation = control_metadata["mouse_observation"]
            command = self._active_mouse_controller().calculate(observation)
        mouse_observation_debug = dict(control_metadata.get("mouse_observation_debug") or {})
        active_aim_y_ratio = self._effective_aim_y_ratio(int(target.cls))
        aim_x = float(mouse_observation_debug.get("predicted_x_px") or 0.0)
        aim_y = float(mouse_observation_debug.get("predicted_y_px") or 0.0)
        pipeline_debug = dict(command.debug)
        pipeline_debug["class_id"] = int(target.cls)
        pipeline_debug["effective_aim_y_ratio"] = active_aim_y_ratio
        predicted_source = bool(getattr(target, "is_predicted", False))
        trajectory_generation = int(
            context.frame_id if context.generation is None else context.generation
        )
        requires_trigger = trigger_mode != "always"
        command_expires_ts_ns = (
            int(context.capture_ts_ns or 0)
            + int(
                float(self._active_dual_phase_config(self.config).freshness_threshold_ms)
                * 1_000_000.0
            )
            if self._is_robust_predictive_active()
            else None
        )
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
            source_track_id=int(getattr(target, "track_id"))
            if hasattr(target, "track_id")
            else None,
            predicted_source=predicted_source,
            trajectory_generation=trajectory_generation,
            trigger_required=(
                requires_trigger if self._is_robust_predictive_active() else None
            ),
            trigger_active=(
                algorithm_trigger_active if self._is_robust_predictive_active() else None
            ),
            command_expires_ts_ns=command_expires_ts_ns,
        )
        output_mode = str(getattr(self.executors, "selected", "kmnet") or "kmnet")
        hardware_kind = "kmnet"
        control_allowed = bool(pipeline_debug.get("control_allowed", True))
        calibration_status = self._calibration_fingerprint_status()
        if calibration_status["control_allowed"] is not True:
            control_allowed = False
        can_emit = (
            (box_input.active or not requires_trigger)
            and trigger_activation_ready
            and control_allowed
        )
        has_movement = int(command.dx) != 0 or int(command.dy) != 0
        will_emit = can_emit and has_movement
        no_send_reason = ""
        if not can_emit:
            no_send_reason = (
                "CONTROL_NOT_ALLOWED"
                if not control_allowed
                else "TRIGGER_INACTIVE"
                if not box_input.active and requires_trigger
                else "TRIGGER_ACTIVATION_DELAY"
            )
        elif not has_movement:
            no_send_reason = (
                command.reason
                if command.reason
                in {
                    "AIM_SETTLED",
                    "ACTUATION_FEEDBACK_PENDING",
                    "ACCUMULATING_FRACTIONAL_COUNTS",
                }
                else "CONTROL_OUTPUT_ZERO"
            )
        trigger_raw = getattr(box_input, "raw", {}) or {}
        trigger_requirement = self._trigger_requirement_label(
            requires_trigger=requires_trigger,
            trigger_mode=trigger_mode,
        )
        control_center_x = float(control_metadata.get("control_width") or 0.0) * 0.5
        control_center_y = float(control_metadata.get("control_height") or 0.0) * 0.5
        aim_error_x = float(aim_x - control_center_x)
        aim_error_y = float(aim_y - control_center_y)
        raw_error_x = float(pipeline_debug.get("observed_error_x_px", aim_error_x))
        raw_error_y = float(pipeline_debug.get("observed_error_y_px", aim_error_y))
        target_detection_index = self._target_detection_index(context, target)
        self.last_target = {
            **self._target_payload(target, context),
            "target_detection_index": target_detection_index,
            "aim_y_ratio": active_aim_y_ratio,
            "aim_x": aim_x,
            "aim_y": aim_y,
            "observed_aim_x": mouse_observation_debug.get("observed_x_px"),
            "observed_aim_y": mouse_observation_debug.get("observed_y_px"),
            "aim_offset_x": float(aim_x - control_center_x),
            "aim_offset_y": float(aim_y - control_center_y),
            "selector_state": selection.state,
            "selection_reason": selection.reason,
            "locked": selection.locked,
            "priority_rank": selection.priority_rank,
            "distance_px": selection.distance_px,
            "quality_score": selection.quality_score,
            "class_score": selection.class_score,
            "distance_score": selection.distance_score,
            "selection_score": selection.selection_score,
            "inside_fov": selection.inside_fov,
            "candidates": selection.candidates,
            "bbox_age_ms": 0.0,
            "is_stale": False,
            "target_key": target_key,
            "frame_age_ms": frame_age_ms,
            "capture_ts_ns": context.capture_ts_ns,
            "candidate_filter": self._candidate_filter_payload(selector_debug),
            "track_diagnostics": track_diagnostics,
            "mouse_observation": mouse_observation_debug,
        }
        self.last_control = {
            "frame_id": context.frame_id,
            "capture_ts_ns": context.capture_ts_ns,
            "control_now_ts_ns": control_now_ns,
            "trajectory_generation": trajectory_generation,
            "frame_age_ms": frame_age_ms,
            "global_state": self._global_state_from_selection_state(selection.state),
            "aim_error_x": aim_error_x,
            "aim_error_y": aim_error_y,
            "raw_error_x": raw_error_x,
            "raw_error_y": raw_error_y,
            "aim_y_ratio": active_aim_y_ratio,
            "aim_x": aim_x,
            "aim_y": aim_y,
            "dx": command.dx,
            "dy": command.dy,
            "confidence": command.confidence,
            "reason": command.reason,
            "pipeline": pipeline_debug,
            "mouse_observation": mouse_observation_debug,
            "selector_state": selection.state,
            "selection_reason": selection.reason,
            "selector_debug": dict(getattr(self.target_selector, "last_debug", {}) or {}),
            "candidate_filter": self._candidate_filter_payload(selector_debug),
            "track_diagnostics": track_diagnostics,
            "target_detection_index": target_detection_index,
            "priority_rank": selection.priority_rank,
            "distance_px": selection.distance_px,
            "quality_score": selection.quality_score,
            "class_score": selection.class_score,
            "distance_score": selection.distance_score,
            "selection_score": selection.selection_score,
            "trigger_active": box_input.active,
            "trigger_required": requires_trigger,
            "trigger_mode": trigger_mode,
            "trigger_hold_ms": trigger_hold_ms,
            "trigger_activation_delay_ms": activation_delay_ms,
            "trigger_activation_ready": trigger_activation_ready,
            "left_trigger_active": bool(hardware_input.left),
            "left_trigger_hold_ms": left_trigger_hold_ms,
            "trigger_requirement": trigger_requirement,
            "trigger_reason": str(
                trigger_raw.get("reason")
                or trigger_raw.get("mode")
                or trigger_raw.get("source")
                or ""
            ),
            "trigger_raw": trigger_raw,
            "output_mode": output_mode,
            "will_emit": will_emit,
            "no_send_reason": no_send_reason,
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
            can_emit=will_emit,
            trigger_raw=trigger_raw,
            output_mode=output_mode,
            hardware_kind=hardware_kind,
            trigger_mode=trigger_mode,
        )
        if not can_emit:
            clear_reason = no_send_reason or "CONTROL_NOT_ALLOWED"
            self._clear_pending_commands(clear_reason)
            # The high-frequency algorithm keeps estimator/mode state while the
            # trigger is released; its quantizer clears itself before returning.
            # The two non-predictive controllers retain their full reset behavior.
            if not self._is_robust_predictive_active():
                self._reset_control_motion_state()
            self.last_execution = {
                "executor_id": str(getattr(self.executors, "selected", "")),
                "sent": False,
                "accepted": False,
                "clipped": False,
                "output_dx": 0.0,
                "output_dy": 0.0,
                "message": (
                    f"触发持续 {trigger_hold_ms:.1f}ms，等待达到 {activation_delay_ms:.1f}ms"
                    if clear_reason == "TRIGGER_ACTIVATION_DELAY"
                    else f"{trigger_requirement}，控制量未发送"
                ),
                "intent": {
                    "dx": float(command.dx),
                    "dy": float(command.dy),
                    "accepted": False,
                    "clipped": False,
                    "reason": command.reason,
                },
            }
            return None
        if not has_movement:
            zero_reason = (
                command.reason
                if command.reason
                in {
                    "AIM_SETTLED",
                    "ACTUATION_FEEDBACK_PENDING",
                    "ACCUMULATING_FRACTIONAL_COUNTS",
                }
                else "CONTROL_OUTPUT_ZERO"
            )
            self._clear_pending_commands(zero_reason)
            self.last_execution = {
                "executor_id": str(getattr(self.executors, "selected", "")),
                "sent": False,
                "accepted": True,
                "clipped": False,
                "output_dx": 0.0,
                "output_dy": 0.0,
                "message": (
                    "瞄点已到位，保持设备静止"
                    if zero_reason == "AIM_SETTLED"
                    else "等待上一条设备输出进入采集画面"
                    if zero_reason == "ACTUATION_FEEDBACK_PENDING"
                    else "累计不足 1 count 的小数余量"
                    if zero_reason == "ACCUMULATING_FRACTIONAL_COUNTS"
                    else "控制量量化为零，未发送设备"
                ),
                "intent": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "accepted": True,
                    "clipped": False,
                    "reason": zero_reason,
                },
            }
            return None
        return intent

    def _clear_pending_commands(self, reason: str) -> None:
        clear = getattr(self.executors, "clear_scheduler", None)
        if callable(clear):
            clear(reason)
            return
        scheduler = getattr(self.executors, "scheduler", None)
        fallback_clear = getattr(scheduler, "clear", None)
        if callable(fallback_clear):
            fallback_clear(reason)

    def _reset_direct_command_executor(self) -> None:
        reset = getattr(self.executors, "reset_mouse_command_executor", None)
        if callable(reset):
            reset()

    def cancel_control(self, reason: str = "RUNTIME_STOPPED") -> None:
        with self._control_lock:
            self.running = False
            self._reset_detection_batch_cursor()
            self._reset_runtime_control_state(reason)

    def reset_runtime_session(self, reason: str = "RUNTIME_STOPPED") -> None:
        """Clear all observation-derived state after the data pipeline is disconnected."""

        with self._control_lock:
            self.running = False
            self._reset_detection_batch_cursor()
            self.last_pipeline_timings = {}
            self.last_inference_reason = reason
            self.last_inference_status = {
                "ran": False,
                "available": False,
                "reason": reason,
                "terminal_rejected": True,
            }
            self.stale_drop_count = 0
            self._reset_runtime_control_state(reason)

    def _record_executed_control(self, result: Any) -> None:
        if not bool(getattr(result, "sent", False)):
            return
        metadata = getattr(result, "metadata", None) or {}
        intent = getattr(result, "intent", None)
        dx = metadata.get("driver_dx", getattr(intent, "dx", 0))
        dy = metadata.get("driver_dy", getattr(intent, "dy", 0))
        send_ts_ns = metadata.get("device_send_end_ts_ns", time.monotonic_ns())
        if not all(isinstance(value, (int, float)) for value in (dx, dy, send_ts_ns)):
            return
        self._executed_control_samples.append((int(send_ts_ns), int(dx), int(dy)))
        self._prune_executed_control_samples(int(send_ts_ns))

    def _executed_control_activity(
        self,
        *,
        control_now_ts_ns: int,
        capture_ts_ns: int,
        measurement_dt_s: float | None,
    ) -> dict[str, Any]:
        self._prune_executed_control_samples(control_now_ts_ns)

        def totals(start_ns: int, end_ns: int) -> tuple[int, int, int, int, int]:
            selected = [
                (dx, dy)
                for send_ts_ns, dx, dy in self._executed_control_samples
                if start_ns < send_ts_ns <= end_ns
            ]
            return (
                sum(dx for dx, _ in selected),
                sum(dy for _, dy in selected),
                sum(abs(dx) + abs(dy) for dx, dy in selected),
                sum(abs(dx) for dx, _ in selected),
                sum(abs(dy) for _, dy in selected),
            )

        recent: dict[int, tuple[int, int, int, int, int]] = {
            window_ms: totals(
                control_now_ts_ns - window_ms * 1_000_000,
                control_now_ts_ns,
            )
            for window_ms in (20, 40, 60)
        }
        if (
            measurement_dt_s is not None
            and math.isfinite(measurement_dt_s)
            and measurement_dt_s > 0.0
        ):
            previous_capture_ts_ns = capture_ts_ns - int(measurement_dt_s * 1e9)
            between_observations = totals(previous_capture_ts_ns, capture_ts_ns)
        else:
            between_observations = (0, 0, 0, 0, 0)
        recent_abs_40 = recent[40][2]
        recent_abs_x_60 = recent[60][3]
        recent_abs_y_60 = recent[60][4]
        latest_send_x_ts_ns = max(
            (send_ts_ns for send_ts_ns, dx, _ in self._executed_control_samples if dx != 0),
            default=0,
        )
        latest_send_y_ts_ns = max(
            (send_ts_ns for send_ts_ns, _, dy in self._executed_control_samples if dy != 0),
            default=0,
        )
        configured_feedback_delay_ns = int(
            max(0.0, float(self.config.control.configured_actuation_delay_s)) * 1e9
        )
        observation_guard_s = (
            min(0.050, float(measurement_dt_s))
            if measurement_dt_s is not None
            and math.isfinite(measurement_dt_s)
            and measurement_dt_s > 0.0
            else 0.0
        )
        observation_guard_ns = int(observation_guard_s * 1e9)
        feedback_wait_ns = configured_feedback_delay_ns + observation_guard_ns
        feedback_visible_after_x_ts_ns = latest_send_x_ts_ns + feedback_wait_ns
        feedback_visible_after_y_ts_ns = latest_send_y_ts_ns + feedback_wait_ns
        actuation_pending_x = bool(
            latest_send_x_ts_ns > 0 and capture_ts_ns <= feedback_visible_after_x_ts_ns
        )
        actuation_pending_y = bool(
            latest_send_y_ts_ns > 0 and capture_ts_ns <= feedback_visible_after_y_ts_ns
        )
        confidence_zero_x = 1
        confidence_zero_y = 1
        velocity_confidence_x = max(
            0.0,
            min(1.0, 1.0 - recent_abs_x_60 / confidence_zero_x),
        )
        velocity_confidence_y = max(
            0.0,
            min(1.0, 1.0 - recent_abs_y_60 / confidence_zero_y),
        )
        velocity_confidence = min(velocity_confidence_x, velocity_confidence_y)
        return {
            "executed_counts_since_previous_observation_x": between_observations[0],
            "executed_counts_since_previous_observation_y": between_observations[1],
            "executed_counts_last_20ms_x": recent[20][0],
            "executed_counts_last_20ms_y": recent[20][1],
            "executed_counts_last_40ms_x": recent[40][0],
            "executed_counts_last_40ms_y": recent[40][1],
            "executed_counts_last_60ms_x": recent[60][0],
            "executed_counts_last_60ms_y": recent[60][1],
            "recent_executed_counts_abs_40ms": recent_abs_40,
            "velocity_confidence": velocity_confidence,
            "velocity_confidence_x": velocity_confidence_x,
            "velocity_confidence_y": velocity_confidence_y,
            "velocity_confidence_zero_counts_x": confidence_zero_x,
            "velocity_confidence_zero_counts_y": confidence_zero_y,
            "velocity_confidence_source": "axis_recent_successful_device_counts_60ms",
            "latest_successful_send_x_ts_ns": latest_send_x_ts_ns,
            "latest_successful_send_y_ts_ns": latest_send_y_ts_ns,
            "feedback_visible_after_x_ts_ns": feedback_visible_after_x_ts_ns,
            "feedback_visible_after_y_ts_ns": feedback_visible_after_y_ts_ns,
            "configured_actuation_feedback_delay_ms": configured_feedback_delay_ns / 1e6,
            "actuation_observation_guard_ms": observation_guard_ns / 1e6,
            "actuation_feedback_delay_ms": feedback_wait_ns / 1e6,
            "actuation_pending_x": actuation_pending_x,
            "actuation_pending_y": actuation_pending_y,
        }

    def _prune_executed_control_samples(self, now_ns: int) -> None:
        cutoff_ns = int(now_ns) - 250_000_000
        while self._executed_control_samples and self._executed_control_samples[0][0] < cutoff_ns:
            self._executed_control_samples.popleft()

    @staticmethod
    def _global_state_from_selection_state(state: Any) -> str:
        normalized = str(state or "").strip().lower()
        if normalized in {
            "fresh",
            "locked",
            "acquire",
            "acquiring",
            "switch_hold",
            "switch_committed",
            "committed_initial",
        }:
            return "TRACKING"
        if normalized in {
            "target_unavailable",
            "missing",
            "no_target",
            "reacquire",
            "lost",
            "switch_pending",
        }:
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
        command_status = (
            str(
                scheduler_execution_payload.get("command_status")
                or scheduler_payload.get("command_status")
                or metadata.get("command_status")
                or ""
            )
            .strip()
            .lower()
        )
        cancel_reason = (
            str(
                scheduler_execution_payload.get("cancel_reason")
                or scheduler_payload.get("cancel_reason")
                or metadata.get("cancel_reason")
                or ""
            )
            .strip()
            .upper()
        )
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
            return "检测到目标后自动控制"
        if trigger_mode == "hardware":
            return "需要 kmNet 硬件按键回传"
        return "需要触发"

    def _log_control_decision(
        self,
        *,
        context: FrameContext,
        target: Track,
        command: Any,
        can_emit: bool,
        trigger_raw: dict[str, Any],
        output_mode: str,
        hardware_kind: str,
        trigger_mode: str,
    ) -> None:
        trigger_source = str(
            trigger_raw.get("source") or trigger_raw.get("mode") or trigger_raw.get("reason") or ""
        )
        target_key = self._control_target_key(target)
        signature = (
            f"target={target_key}|emit={can_emit}|out={output_mode}|hardware={hardware_kind}|"
            f"trigger={trigger_mode}|source={trigger_source}|"
            f"active={bool(trigger_raw.get('left') or trigger_raw.get('right') or trigger_raw.get('active'))}|"
            f"reason={str(getattr(command, 'reason', ''))}"
        )
        if signature == self._last_control_log_signature:
            return
        self._last_control_log_signature = signature
        self._last_no_target_log_signature = ""
        logger.debug(
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
        selector_debug = dict(getattr(self.target_selector, "last_debug", {}) or {})
        diagnostics = self._target_pipeline_diagnostics(
            selection=selection,
            selector_debug=selector_debug,
        )
        signature = "|".join(
            (
                str(diagnostics["code"]),
                str(diagnostics["stage"]),
                ",".join(diagnostics["rejection_reasons"]),
            )
        )
        if signature == self._last_no_target_log_signature:
            return
        self._last_no_target_log_signature = signature
        self._last_control_log_signature = ""
        logger.warning(
            "target pipeline blocked frame=%s code=%s stage=%s message=%s counts=%s "
            "rejection_reasons=%s",
            context.frame_id,
            diagnostics["code"],
            diagnostics["stage"],
            diagnostics["message"],
            diagnostics["counts"],
            diagnostics["rejection_reasons"],
        )

    def _log_box_input_state(self, state: BoxInputState, source: str) -> None:
        if not state.active:
            self._last_box_input_log_signature = ""
            return
        raw = getattr(state, "raw", {}) or {}
        signature = (
            f"box|source={source}|active={state.active}|left={state.left}|"
            f"right={state.right}|side={state.side}"
        )
        if signature == self._last_box_input_log_signature:
            return
        self._last_box_input_log_signature = signature
        logger.info(
            "box input source=%s active=%s left=%s right=%s side=%s raw=%s",
            source,
            state.active,
            state.left,
            state.right,
            state.side,
            raw,
        )

    def _select_control_target(self, context: FrameContext) -> TargetSelection:
        fov_ratio = max(
            0.0,
            min(1.0, float(self.config.control.target_fov_radius_px) / 640.0),
        )
        control_center_x_px, control_center_y_px = self._control_center_in_roi(context)
        return self.target_selector.select(
            context,
            min_confidence=float(self.config.inference.confidence_threshold),
            fov_ratio=fov_ratio,
            aim_ratio=self._active_aim_ratio(),
            class_aim_y_ratios=self._active_class_aim_y_ratios(),
            control_center_x_px=control_center_x_px,
            control_center_y_px=control_center_y_px,
            class_filter=str(getattr(self.config.inference, "detection_class_filter", "all")),
            class_priority=self._class_priority(),
            sticky_bias=float(getattr(self.config.control, "target_sticky_bias", 0.25)),
            lock_enabled=bool(getattr(self.config.control, "target_lock_enabled", True)),
            lost_grace_frames=0,
            ratio_max_aspect=float(getattr(self.config.control, "candidate_ratio_max_aspect", 6.0)),
            quality_confidence_weight=float(
                getattr(self.config.control, "candidate_quality_confidence_weight", 0.7)
            ),
            quality_area_weight=float(
                getattr(self.config.control, "candidate_quality_area_weight", 0.3)
            ),
            selection_class_weight=float(
                getattr(self.config.control, "candidate_selection_class_weight", 0.55)
            ),
            selection_quality_weight=float(
                getattr(self.config.control, "candidate_selection_quality_weight", 0.05)
            ),
            selection_distance_weight=float(
                getattr(self.config.control, "candidate_selection_distance_weight", 0.40)
            ),
            tracker_max_match_distance=float(
                getattr(self.config.control, "tracker_max_match_distance", 1.5)
            ),
            tracker_position_cost_weight=float(
                getattr(self.config.control, "tracker_position_cost_weight", 0.75)
            ),
            tracker_iou_cost_weight=float(
                getattr(self.config.control, "tracker_iou_cost_weight", 0.25)
            ),
            tracker_max_missed_frames=int(
                getattr(self.config.control, "tracker_max_missed_frames", 2)
            ),
            target_switch_min_preference_advantage=float(
                getattr(self.config.control, "target_switch_min_preference_advantage", 0.08)
            ),
            target_switch_min_continuity_score=float(
                getattr(self.config.control, "target_switch_min_continuity_score", 0.70)
            ),
            target_switch_delay_ms=float(self.config.control.target_switch_delay_ms),
            kalman_acceleration_noise=float(
                getattr(self.config.control, "kalman_acceleration_noise", 1200.0)
            ),
            kalman_measurement_noise_x=float(
                getattr(self.config.control, "kalman_measurement_noise_x", 16.0)
            ),
            kalman_measurement_noise_y=float(
                getattr(self.config.control, "kalman_measurement_noise_y", 16.0)
            ),
            kalman_max_predict_missing_ms=float(
                getattr(self.config.control, "kalman_max_predict_missing_ms", 80.0)
            ),
            kalman_max_predict_steps=int(
                getattr(self.config.control, "kalman_max_predict_steps", 5)
            ),
            kalman_max_predict_dt_ms=float(
                getattr(self.config.control, "kalman_max_predict_dt_ms", 35.0)
            ),
            kalman_max_position_sigma_px=float(
                getattr(self.config.control, "kalman_max_position_sigma_px", 45.0)
            ),
            kalman_max_covariance_trace=float(
                getattr(self.config.control, "kalman_max_covariance_trace", 5000.0)
            ),
            kalman_nis_threshold=float(getattr(self.config.control, "kalman_nis_threshold", 9.21)),
            kalman_nis_hard_reject=float(
                getattr(self.config.control, "kalman_nis_hard_reject", 16.0)
            ),
            kalman_min_identity_confidence=float(
                getattr(self.config.control, "kalman_min_identity_confidence", 0.70)
            ),
            kalman_min_prediction_confidence=float(
                getattr(self.config.control, "kalman_min_prediction_confidence", 0.35)
            ),
            kalman_prediction_decay_tau_ms=float(
                getattr(self.config.control, "kalman_prediction_decay_tau_ms", 45.0)
            ),
        )

    def _control_center_in_roi(self, context: FrameContext) -> tuple[float, float]:
        if self.last_inference_status.get("source_geometry_trusted") is True:
            source_width = self._status_int("source_width", 0)
            source_height = self._status_int("source_height", 0)
            if source_width > 0 and source_height > 0:
                return (
                    source_width * 0.5 - self._status_int("roi_offset_x", 0),
                    source_height * 0.5 - self._status_int("roi_offset_y", 0),
                )
        return context.width * 0.5, context.height * 0.5

    def _active_aim_ratio(self) -> float:
        return float(self.config.control.aim.role_y_ratios.other)

    def _active_class_aim_y_ratios(self) -> dict[int, float]:
        profile_name = str(
            getattr(self.config.inference, "detection_class_profile", "default")
        )
        by_profile = getattr(self.config.control.aim, "class_roles", {}) or {}
        raw_roles = by_profile.get(profile_name, {})
        ratios = self.config.control.aim.role_y_ratios
        return {
            int(class_id): float(getattr(ratios, str(role), ratios.other))
            for class_id, role in raw_roles.items()
        }

    def _effective_aim_y_ratio(self, class_id: int) -> float:
        return resolve_aim_y_ratio(
            self._active_aim_ratio(),
            self._active_class_aim_y_ratios(),
            int(class_id),
        )

    def _filter_detections_by_config(self, detections: list[Detection]) -> list[Detection]:
        allowed = parse_allowed_class_ids(
            str(getattr(self.config.inference, "detection_class_filter", "all"))
        )
        if allowed is None:
            return detections
        return [item for item in detections if int(item.cls) in allowed]

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
        button_reader = getattr(self.executors, "read_buttons", None)
        buttons: dict[str, Any] | None = None
        if callable(button_reader):
            try:
                buttons = button_reader()
            except Exception as exc:
                buttons = {
                    "available": False,
                    "left": False,
                    "right": False,
                    "reason": f"button reader exception: {exc}",
                }
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

    @staticmethod
    def _box_input_hold_ms(
        state: BoxInputState,
        now_ns: int,
        *,
        left_only: bool = False,
    ) -> float:
        raw = state.raw if isinstance(state.raw, dict) else {}
        starts: list[int] = []
        if state.left:
            starts.append(int(raw.get("left_pressed_since_ts_ns") or 0))
        if not left_only and state.right:
            starts.append(int(raw.get("right_pressed_since_ts_ns") or 0))
        starts = [value for value in starts if value > 0]
        if not starts:
            return 0.0
        return max(0.0, (int(now_ns) - min(starts)) / 1e6)

    def _create_control_algorithm_registry(self, config: RuntimeConfig) -> AlgorithmRegistry:
        algorithm_id = str(config.control.active_algorithm)
        controller = (
            self._create_dual_phase_algorithm(config)
            if algorithm_id == DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2
            else self._create_mouse_controller(config)
        )
        return AlgorithmRegistry(algorithm_id, controller)

    def _create_mouse_controller(self, config: RuntimeConfig) -> MouseController:
        max_plan_steps = plan_step_capacity(config.control.scheduler_interval_ms)
        shared = config.control.shared
        algorithm_id = str(config.control.active_algorithm)
        calibrated_config: CalibratedAngularControllerConfig | None = None
        universal_config: UniversalSaturatedControllerConfig | None = None
        if algorithm_id == CALIBRATED_ANGULAR:
            calibrated = config.control.calibrated_angular
            calibrated_config = CalibratedAngularControllerConfig(
                fov_x_deg=float(calibrated.fov_x_deg),
                counts_per_360_x=float(calibrated.counts_per_360_x),
                counts_per_360_y=float(calibrated.counts_per_360_y),
                kp_x=float(calibrated.kp_x),
                kp_y=float(calibrated.kp_y),
                kd_x=float(calibrated.kd_x),
                kd_y=float(calibrated.kd_y),
                d_ema_alpha=float(calibrated.d_ema_alpha),
                max_angle_step_x_rad=math.radians(float(calibrated.max_angle_step_x_deg)),
                max_angle_step_y_rad=math.radians(float(calibrated.max_angle_step_y_deg)),
            )
        elif algorithm_id == UNIVERSAL_SATURATED:
            universal = config.control.universal_saturated
            universal_config = UniversalSaturatedControllerConfig(
                response_scale_x_px=float(universal.response_scale_x_px),
                response_scale_y_px=float(universal.response_scale_y_px),
                max_step_x_counts=float(universal.max_step_x_counts),
                max_step_y_counts=float(universal.max_step_y_counts),
            )
        else:
            raise ValueError(f"unsupported mouse control algorithm: {algorithm_id}")
        return MouseController(
            MouseControllerConfig(
                mode=algorithm_id,
                shared=SharedOutputConfig(
                    deadzone_x_px=float(shared.deadzone_x_px),
                    deadzone_y_px=float(shared.deadzone_y_px),
                    max_count_slew_x=float(shared.max_count_slew_x),
                    max_count_slew_y=float(shared.max_count_slew_y),
                    invert_y=bool(shared.invert_y),
                    max_budget_counts_x=int(config.control.scheduler_step_counts_x)
                    * max_plan_steps,
                    max_budget_counts_y=int(config.control.scheduler_step_counts_y)
                    * max_plan_steps,
                    recoil_enabled=bool(shared.recoil_enabled),
                    recoil_start_delay_ms=float(shared.recoil_start_delay_ms),
                    recoil_y_rate_counts_s=float(shared.recoil_y_rate_counts_s),
                    recoil_ramp_up_ms=float(shared.recoil_ramp_up_ms),
                    recoil_max_counts_per_observation=float(
                        shared.recoil_max_counts_per_observation
                    ),
                ),
                calibrated_angular=calibrated_config,
                universal_saturated=universal_config,
            )
        )

    @staticmethod
    def _create_dual_phase_algorithm(
        config: RuntimeConfig,
    ) -> DualPhaseAtanRobustPredictiveV2Algorithm:
        source_v2 = config.control.dual_phase_atan_robust_predictive_v2
        return DualPhaseAtanRobustPredictiveV2Algorithm(
            DualPhaseRobustAlgorithmConfig(
                freshness_threshold_ms=float(source_v2.freshness_threshold_ms),
                projection=DualPhaseRobustProjectionConfig(
                    fov_x_deg=float(source_v2.projection.fov_x_deg),
                    counts_per_360=float(source_v2.projection.counts_per_360),
                    invert_y=bool(source_v2.projection.invert_y),
                ),
                mode=DualPhaseRobustModeSelectorConfig(
                    near_threshold_px=float(source_v2.mode.near_threshold_px),
                ),
                velocity=DualPhaseRobustVelocityConfig(
                    history_size=int(source_v2.velocity.history_size),
                    velocity_sample_count=int(source_v2.velocity.velocity_sample_count),
                    smoothing_frames=float(source_v2.velocity.smoothing_frames),
                    history_reset_gap_ms=float(source_v2.velocity.history_reset_gap_ms),
                    spread_base_px_ms=float(source_v2.velocity.spread_base_px_ms),
                    spread_relative=float(source_v2.velocity.spread_relative),
                    change_base_px_ms=float(source_v2.velocity.change_base_px_ms),
                    change_relative=float(source_v2.velocity.change_relative),
                ),
                prediction=DualPhaseRobustPredictionConfig(
                    lead_frames=float(source_v2.prediction.lead_frames),
                    far=DualPhaseRobustPredictionModeConfig(
                        absolute_cap_px=float(source_v2.prediction.far.absolute_cap_px),
                        base_cap_px=float(source_v2.prediction.far.base_cap_px),
                        relative_cap=float(source_v2.prediction.far.relative_cap),
                    ),
                    near=DualPhaseRobustPredictionModeConfig(
                        absolute_cap_px=float(source_v2.prediction.near.absolute_cap_px),
                        base_cap_px=float(source_v2.prediction.near.base_cap_px),
                        relative_cap=float(source_v2.prediction.near.relative_cap),
                    ),
                ),
                recoil=DualPhaseRobustRecoilConfig(
                    enabled=bool(config.control.shared.recoil_enabled),
                    start_delay_ms=float(config.control.shared.recoil_start_delay_ms),
                    y_rate_counts_s=float(config.control.shared.recoil_y_rate_counts_s),
                    ramp_up_ms=float(config.control.shared.recoil_ramp_up_ms),
                    max_counts_per_observation=float(
                        config.control.shared.recoil_max_counts_per_observation
                    ),
                ),
                atan=DualPhaseRobustAtanControllerConfig(
                    scale_counts=float(source_v2.atan.scale_counts),
                    far=DualPhaseRobustAtanModeConfig(
                        kp=float(source_v2.atan.far.kp),
                        max_counts_per_update=float(source_v2.atan.far.max_counts_per_update),
                    ),
                    near=DualPhaseRobustAtanModeConfig(
                        kp=float(source_v2.atan.near.kp),
                        max_counts_per_update=float(source_v2.atan.near.max_counts_per_update),
                    ),
                ),
            )
        )

    def _reset_control_motion_state(self) -> None:
        self.control_algorithms.reset()

    @staticmethod
    def _target_detection_index(context: FrameContext, target: Track) -> int | None:
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

    @staticmethod
    def _control_target_key(target: Track) -> str:
        return f"track:{int(target.track_id)}"

    def _dual_phase_control_command(
        self,
        *,
        context: FrameContext,
        target: Track,
        control_metadata: dict[str, Any],
        selector_debug: dict[str, Any],
        control_now_ts_ns: int,
        trigger_active: bool,
        left_trigger_active: bool,
        left_trigger_hold_ms: float,
        measurement_dt_ms: float | None,
    ) -> tuple[MoveCommand, dict[str, Any]]:
        algorithm_id = DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2
        transform = self._coordinate_transform_for_context(context)
        control_width = float(control_metadata.get("control_width") or 0.0)
        control_height = float(control_metadata.get("control_height") or 0.0)
        raw_aim = self.raw_aim_projector.project(
            track=target,
            frame_id=context.frame_id,
            capture_ts_ns=int(context.capture_ts_ns or 0),
            y_ratio=self._effective_aim_y_ratio(int(target.cls)),
            coordinate_transform=transform,
            control_width_px=control_width,
            control_height_px=control_height,
            source_geometry_trusted=(control_metadata.get("capture_geometry_trusted") is True),
        )
        if transform is None:
            crosshair_roi_x = 0.0
            crosshair_roi_y = 0.0
            source_width = 0
            source_height = 0
            roi_width = 0
            roi_height = 0
            roi_left = 0
            roi_top = 0
        else:
            crosshair_capture = transform.control_to_capture_point(
                control_width * 0.5,
                control_height * 0.5,
            )
            crosshair_roi = transform.capture_to_roi_point(
                crosshair_capture.x,
                crosshair_capture.y,
            )
            crosshair_roi_x = float(crosshair_roi.x)
            crosshair_roi_y = float(crosshair_roi.y)
            source_width = int(round(transform.capture_width))
            source_height = int(round(transform.capture_height))
            roi_width = int(round(transform.roi_width))
            roi_height = int(round(transform.roi_height))
            roi_left = int(round(transform.roi_x))
            roi_top = int(round(transform.roi_y))

        motion = target_motion_estimate_from_debug(
            track=target,
            capture_ts_ns=context.capture_ts_ns,
            tracker_debug=selector_debug,
        )
        track_confidence = (
            float(motion.identity_confidence)
            if motion.valid and math.isfinite(float(motion.identity_confidence))
            else 0.0
        )
        observation_kwargs = {
            "generation": int(
                context.frame_id if context.generation is None else context.generation
            ),
            "frame_id": int(context.frame_id),
            "target_id": int(target.track_id),
            "capture_ts_ns": int(context.capture_ts_ns or 0),
            "inference_end_ts_ns": int(context.inference_end_ts_ns or control_now_ts_ns),
            "control_now_ns": int(control_now_ts_ns),
            "aim_x": float(raw_aim.aim_roi_x_px),
            "aim_y": float(raw_aim.aim_roi_y_px),
            "crosshair_x": crosshair_roi_x,
            "crosshair_y": crosshair_roi_y,
            "bbox_x1": float(target.x1),
            "bbox_y1": float(target.y1),
            "bbox_x2": float(target.x2),
            "bbox_y2": float(target.y2),
            "observation_width": int(context.width),
            "observation_height": int(context.height),
            "roi_left": roi_left,
            "roi_top": roi_top,
            "roi_width": roi_width,
            "roi_height": roi_height,
            "source_width": source_width,
            "source_height": source_height,
            "detection_confidence": float(target.score),
            "track_confidence": max(0.0, min(1.0, track_confidence)),
            "trigger_active": bool(trigger_active),
            "left_trigger_active": bool(left_trigger_active),
            "left_trigger_hold_ms": float(left_trigger_hold_ms),
            "measurement_dt_ms": measurement_dt_ms,
            "target_valid": bool(
                raw_aim.valid
                and not getattr(target, "is_predicted", False)
                and not getattr(target, "is_stale", False)
            ),
        }
        tracker_debug = selector_debug.get("tracker") if isinstance(selector_debug, dict) else None
        if not isinstance(tracker_debug, dict):
            tracker_debug = {}
        rebuilt_track_ids = {
            int(track_id)
            for key in ("created_track_ids", "restored_track_ids")
            for track_id in tracker_debug.get(key, [])
            if isinstance(track_id, int)
        }
        observation_kwargs["track_rebuilt"] = int(target.track_id) in rebuilt_track_ids
        observation = DualPhaseAtanRobustPredictiveV2Observation(**observation_kwargs)
        decision = self._active_robust_predictive_controller().calculate(observation)
        telemetry = {
            **decision.telemetry,
            "capture_ts_ns": observation.capture_ts_ns,
            "inference_end_ts_ns": observation.inference_end_ts_ns,
            "control_now_ns": observation.control_now_ns,
            "aim_x": observation.aim_x,
            "aim_y": observation.aim_y,
            "bbox_x1": observation.bbox_x1,
            "bbox_y1": observation.bbox_y1,
            "bbox_x2": observation.bbox_x2,
            "bbox_y2": observation.bbox_y2,
            "bbox_width": max(0.0, observation.bbox_x2 - observation.bbox_x1),
            "bbox_height": max(0.0, observation.bbox_y2 - observation.bbox_y1),
            "detection_confidence": observation.detection_confidence,
            "track_confidence": observation.track_confidence,
            "delivery_mode": "latest_replace",
            "scheduler_used": True,
        }
        block_reason = str(decision.block_reason or "")
        control_allowed = block_reason in {"", "TRIGGER_INACTIVE"}
        reason = (
            block_reason
            if block_reason
            else algorithm_id
            if decision.dx != 0 or decision.dy != 0
            else "ACCUMULATING_FRACTIONAL_COUNTS"
        )
        debug = {
            **telemetry,
            "algorithm": algorithm_id,
            "control_mode": telemetry.get("mode", "far"),
            "control_allowed": control_allowed,
            "observed_error_x_px": float(telemetry.get("error_real_x") or 0.0),
            "observed_error_y_px": float(telemetry.get("error_real_y") or 0.0),
            "predicted_error_x_px": float(telemetry.get("error_control_x") or 0.0),
            "predicted_error_y_px": float(telemetry.get("error_control_y") or 0.0),
        }
        source_prediction_offset_x = float(telemetry.get("prediction_safe_offset_x") or 0.0) * (
            roi_width / max(1, int(context.width))
        )
        mouse_observation_debug = {
            "algorithm_id": algorithm_id,
            "observed_x_px": float(raw_aim.aim_control_x_px),
            "observed_y_px": float(raw_aim.aim_control_y_px),
            "predicted_x_px": float(raw_aim.aim_control_x_px) + source_prediction_offset_x,
            "predicted_y_px": float(raw_aim.aim_control_y_px),
            "observed_x_roi_px": observation.aim_x,
            "observed_y_roi_px": observation.aim_y,
            "crosshair_x_roi_px": observation.crosshair_x,
            "crosshair_y_roi_px": observation.crosshair_y,
            "valid": observation.target_valid,
            "invalid_reason": raw_aim.invalid_reason,
            "raw_aim": raw_aim.debug_payload(),
            "telemetry": telemetry,
        }
        command = MoveCommand(
            dx=int(decision.dx),
            dy=int(decision.dy),
            confidence=float(target.score),
            reason=reason,
            debug=debug,
        )
        return command, {
            "mouse_observation": observation,
            "mouse_observation_debug": mouse_observation_debug,
        }

    def _mouse_observation_metadata(
        self,
        *,
        context: FrameContext,
        target: Track,
        control_metadata: dict[str, Any],
        selector_debug: dict[str, Any],
        control_now_ts_ns: int,
        measurement_dt_s: float | None,
        left_trigger_active: bool,
        left_trigger_hold_ms: float,
    ) -> dict[str, Any]:
        transform = self._coordinate_transform_for_context(context)
        control_width = float(control_metadata.get("control_width") or 0.0)
        control_height = float(control_metadata.get("control_height") or 0.0)
        source_geometry_trusted = control_metadata.get("capture_geometry_trusted") is True
        y_ratio = self._effective_aim_y_ratio(int(target.cls))
        raw_aim = self.raw_aim_projector.project(
            track=target,
            frame_id=context.frame_id,
            capture_ts_ns=context.capture_ts_ns,
            y_ratio=y_ratio,
            coordinate_transform=transform,
            control_width_px=control_width,
            control_height_px=control_height,
            source_geometry_trusted=source_geometry_trusted,
        )
        capture_ts_ns = int(context.capture_ts_ns or 0)
        executed_control = self._executed_control_activity(
            control_now_ts_ns=control_now_ts_ns,
            capture_ts_ns=capture_ts_ns,
            measurement_dt_s=measurement_dt_s,
        )
        observed_roi_x = float(raw_aim.aim_roi_x_px)
        observed_roi_y = float(raw_aim.aim_roi_y_px)
        predicted_roi_x = observed_roi_x
        predicted_roi_y = observed_roi_y
        predicted_control_x = float(raw_aim.aim_control_x_px)
        predicted_control_y = float(raw_aim.aim_control_y_px)
        prediction_valid = raw_aim.valid
        invalid_reason = raw_aim.invalid_reason

        observation = MouseObservation(
            frame_id=context.frame_id,
            target_id=int(target.track_id),
            capture_ts_ns=int(context.capture_ts_ns or 0),
            control_now_ts_ns=control_now_ts_ns,
            measurement_dt_s=measurement_dt_s,
            control_width_px=control_width,
            control_height_px=control_height,
            observed_x_px=raw_aim.aim_control_x_px,
            observed_y_px=raw_aim.aim_control_y_px,
            predicted_x_px=predicted_control_x,
            predicted_y_px=predicted_control_y,
            prediction_horizon_s=0.0,
            target_confidence=max(0.0, min(1.0, float(target.score))),
            prediction_confidence=0.0,
            observed_valid=raw_aim.valid and not bool(target.is_predicted),
            actuation_pending_x=bool(executed_control["actuation_pending_x"]),
            actuation_pending_y=bool(executed_control["actuation_pending_y"]),
            left_trigger_active=bool(left_trigger_active),
            left_trigger_hold_ms=max(0.0, float(left_trigger_hold_ms)),
            valid=prediction_valid,
            invalid_reason=invalid_reason,
        )
        return {
            "mouse_observation": observation,
            "mouse_observation_debug": {
                **asdict(observation),
                "prediction_source": "none",
                "prediction_strength": 0.0,
                **executed_control,
                "prediction_origin_x_roi_px": observed_roi_x,
                "prediction_origin_y_roi_px": observed_roi_y,
                "prediction_origin_source_x": "observed",
                "prediction_origin_source_y": "observed",
                "predicted_aim_x_roi_px": predicted_roi_x,
                "predicted_aim_y_roi_px": predicted_roi_y,
                "raw_aim": raw_aim.debug_payload(),
            },
        }

    def _control_frame_metadata(self, context: FrameContext) -> dict[str, Any]:
        inference_debug = self.last_inference_status.get("debug", {})
        preprocess = (
            inference_debug.get("preprocess", {}) if isinstance(inference_debug, dict) else {}
        )
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
            "model_height": preprocess.get("model_height")
            if isinstance(preprocess, dict)
            else None,
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
        tracked_filter = selector_debug.get("tracked_filter")
        basic_filter = selector_debug.get("basic_filter")
        association_filter = selector_debug.get("association_filter")
        effective = tracked_filter if isinstance(tracked_filter, dict) else selector_debug
        candidates = effective.get("candidates")
        rejected = effective.get("rejected")
        selected = selector_debug.get("selected")
        basic_rejected = (
            cls._debug_dict_list(basic_filter.get("rejected"))
            if isinstance(basic_filter, dict)
            else cls._debug_dict_list(selector_debug.get("rejected"))
        )
        rejected_class_ids = sorted(
            {
                int(item["class_id"])
                for item in basic_rejected
                if item.get("reason") == "class_filter"
                and isinstance(item.get("class_id"), int)
            }
        )
        return {
            "effective_class_filter": str(selector_debug.get("class_filter") or "all"),
            "raw_candidates": cls._debug_int(selector_debug, "raw_candidates"),
            "basic_filtered_candidates": cls._debug_int(selector_debug, "filtered_candidates"),
            "filtered_candidates": cls._debug_int(effective, "filtered_candidates"),
            "inside_fov": cls._debug_int(effective, "inside_fov"),
            "rejected_candidates": cls._debug_int(effective, "rejected_candidates"),
            "candidates": cls._debug_dict_list(candidates),
            "rejected": cls._debug_dict_list(rejected),
            "selected": dict(selected) if isinstance(selected, dict) else None,
            "reason": str(selector_debug.get("reason") or ""),
            "selection_center_px": (
                dict(selector_debug["control_center_roi_px"])
                if isinstance(selector_debug.get("control_center_roi_px"), dict)
                else None
            ),
            "selection_radius_px": selector_debug.get("fov_radius_px"),
            "basic": {
                "raw_candidates": cls._debug_int(basic_filter, "raw_candidates")
                if isinstance(basic_filter, dict)
                else cls._debug_int(selector_debug, "raw_candidates"),
                "filtered_candidates": cls._debug_int(basic_filter, "filtered_candidates")
                if isinstance(basic_filter, dict)
                else cls._debug_int(selector_debug, "filtered_candidates"),
                "rejected_candidates": cls._debug_int(basic_filter, "rejected_candidates")
                if isinstance(basic_filter, dict)
                else cls._debug_int(selector_debug, "rejected_candidates"),
                "rejected": basic_rejected,
                "rejected_class_ids": rejected_class_ids,
            },
            "association": dict(association_filter)
            if isinstance(association_filter, dict)
            else None,
            "tracked": dict(tracked_filter) if isinstance(tracked_filter, dict) else None,
        }

    def _target_pipeline_diagnostics(
        self,
        *,
        selection: TargetSelection | None = None,
        selector_debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        inference = dict(self.last_inference_status or {})
        debug = inference.get("debug")
        debug = debug if isinstance(debug, dict) else {}
        decode = debug.get("decode")
        decode = decode if isinstance(decode, dict) else {}
        control = dict(self.last_control or {})
        current_selector_debug = selector_debug
        if current_selector_debug is None:
            stored_selector = control.get("selector_debug")
            current_selector_debug = stored_selector if isinstance(stored_selector, dict) else {}
        candidate = self._candidate_filter_payload(current_selector_debug)
        tracker = current_selector_debug.get("tracker")
        tracker = tracker if isinstance(tracker, dict) else {}
        rejection_reasons = sorted(
            {
                str(item.get("reason"))
                for item in candidate["rejected"]
                if isinstance(item, dict) and item.get("reason")
            }
        )
        counts = {
            "decode_raw_candidates": self._debug_int(decode, "raw_candidates"),
            "threshold_candidates": self._debug_int(decode, "threshold_candidates"),
            "nms_detections": self._debug_int(decode, "nms_detections"),
            "raw_detections": self._debug_int(inference, "raw_detections"),
            "mapped_detections": self._debug_int(inference, "mapped_detections"),
            "basic_candidates": int(candidate["basic"]["filtered_candidates"]),
            "association_candidates": (
                self._debug_int(candidate["association"], "filtered_candidates")
                if isinstance(candidate.get("association"), dict)
                else int(candidate["filtered_candidates"])
            ),
            "tracker_active": self._debug_int(tracker, "active_tracks"),
            "tracker_tentative": self._debug_int(tracker, "tentative_tracks"),
            "tracker_lost": self._debug_int(tracker, "lost_track_count"),
            "filtered_candidates": int(candidate["filtered_candidates"]),
            "inside_fov": int(candidate["inside_fov"]),
        }
        selection_reason = (
            str(selection.reason)
            if selection is not None
            else str(control.get("selection_reason") or candidate["reason"] or "")
        )

        if inference.get("ran") is not True and inference.get("source") != "detection_batch":
            code, stage, message = (
                "INFERENCE_NOT_RUN",
                "inference",
                str(inference.get("reason") or "inference has not run"),
            )
        elif inference.get("available") is not True:
            code, stage, message = (
                "INFERENCE_UNAVAILABLE",
                "inference",
                str(inference.get("reason") or "inference result is unavailable"),
            )
        elif counts["mapped_detections"] <= 0:
            if decode and counts["decode_raw_candidates"] <= 0:
                code, stage, message = (
                    "DECODE_EMPTY_OUTPUT",
                    "inference_decode",
                    str(decode.get("reason") or "model output contains no decode candidates"),
                )
            elif decode and counts["threshold_candidates"] <= 0:
                max_score = decode.get("max_score")
                code, stage, message = (
                    "CONFIDENCE_THRESHOLD_REJECTED",
                    "inference_decode",
                    f"no model candidate passed confidence threshold; max_score={max_score}",
                )
            elif decode and counts["nms_detections"] <= 0:
                code, stage, message = (
                    "NMS_EMPTY",
                    "inference_decode",
                    str(decode.get("reason") or "no detection remained after NMS"),
                )
            else:
                code, stage, message = (
                    "NO_MAPPED_DETECTIONS",
                    "detection_mapping",
                    str(inference.get("reason") or "inference produced no mapped detections"),
                )
        elif counts["basic_candidates"] <= 0:
            basic_rejections = candidate["basic"]["rejected"]
            rejection_reasons = sorted(
                {
                    str(item.get("reason"))
                    for item in basic_rejections
                    if isinstance(item, dict) and item.get("reason")
                }
            )
            code, stage, message = (
                "BASIC_CANDIDATE_REJECTED",
                "basic_filter",
                selection_reason
                or "all detections were rejected by class, confidence, or bbox validation",
            )
        elif counts["association_candidates"] <= 0:
            reason_codes = set(rejection_reasons)
            if reason_codes == {"selection_fov"}:
                code = "OUTSIDE_TARGET_FOV"
            elif reason_codes == {"ratio_check"}:
                code = "BBOX_RATIO_REJECTED"
            else:
                code = "ASSOCIATION_FILTER_REJECTED"
            stage, message = (
                "association_filter",
                selection_reason or "all detections were rejected before association",
            )
        elif counts["tracker_active"] <= 0:
            code = "TRACKER_ACQUIRING" if counts["tracker_tentative"] > 0 else "TRACKER_NO_ACTIVE"
            stage = "tracker"
            message = str(
                tracker.get("reason") or selection_reason or "tracker produced no CONFIRMED track"
            )
        elif counts["filtered_candidates"] <= 0 or counts["inside_fov"] <= 0:
            reason_codes = set(rejection_reasons)
            if reason_codes == {"selection_fov"}:
                code = "OUTSIDE_TARGET_FOV"
            elif reason_codes == {"confidence_filter"}:
                code = "CONTROL_CONFIDENCE_REJECTED"
            elif reason_codes == {"class_filter"}:
                code = "CONTROL_CLASS_REJECTED"
            elif reason_codes == {"ratio_check"}:
                code = "BBOX_RATIO_REJECTED"
            elif reason_codes == {"invalid_bbox"}:
                code = "INVALID_BBOX"
            else:
                code = "TARGET_FILTER_REJECTED"
            stage, message = "target_filter", selection_reason or "all ACTIVE tracks were rejected"
        elif self.last_target is not None or (
            selection is not None and selection.target is not None
        ):
            code, stage, message = (
                "TARGET_SELECTED",
                "selected",
                selection_reason or "target selected",
            )
        else:
            selector_state = str(
                selection.state if selection is not None else control.get("selector_state") or ""
            )
            code = (
                "TARGET_SWITCH_PENDING"
                if selector_state == "switch_pending"
                else "TARGET_UNAVAILABLE"
            )
            stage, message = "target_selection", selection_reason or "no target selected"

        return {
            "code": code,
            "stage": stage,
            "message": message,
            "selection_reason": selection_reason,
            "rejection_reasons": rejection_reasons,
            "effective_class_filter": candidate["effective_class_filter"],
            "rejected_class_ids": candidate["basic"]["rejected_class_ids"],
            "counts": counts,
        }

    def _track_diagnostics_payload(
        self,
        selector_debug: dict[str, Any],
        *,
        target: Track | None = None,
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
            else selection_state
            if selected_track_id is not None
            else ""
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
            "identity_uncertain_tracks": self._debug_int_list(
                tracker.get("identity_uncertain_tracks")
            ),
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
    def _find_track_debug(
        tracks: list[dict[str, Any]], track_id: int | None
    ) -> dict[str, Any] | None:
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

    def _target_payload(self, target: Track, context: FrameContext) -> dict[str, Any]:
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
        roi_offset_x = (
            int(transform.roi_x) if transform is not None else self._status_int("roi_offset_x", 0)
        )
        roi_offset_y = (
            int(transform.roi_y) if transform is not None else self._status_int("roi_offset_y", 0)
        )
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

    def _coordinate_transform_for_context(
        self, context: FrameContext
    ) -> CoordinateTransform | None:
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
    def _coordinate_transform_payload(
        transform: CoordinateTransform | None,
    ) -> dict[str, float] | None:
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
                "target_pipeline": self._target_pipeline_diagnostics(),
                "trace": self._business_trace(None),
            }
        return {
            "frame_id": context.frame_id,
            "detections": len(context.detections),
            "detection_items": [
                self._detection_payload(item, context) for item in context.detections
            ],
            "tracks": len(context.tracks),
            "classes": list(context.classes),
            "inference_reason": self.last_inference_reason,
            "inference": dict(self.last_inference_status),
            "calibration": self._calibration_fingerprint_status(),
            "target": self.last_target,
            "control": self.last_control,
            "execution": self.last_execution,
            "target_pipeline": self._target_pipeline_diagnostics(),
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
        return {
            "id": "capture",
            "label": "采集",
            "status": status,
            "message": message,
            "detail": detail,
        }

    @staticmethod
    def _trace_roi_stage(inference: dict[str, Any]) -> dict[str, Any]:
        if not inference:
            return {
                "id": "roi",
                "label": "ROI",
                "status": "blocked",
                "message": "等待采集帧",
                "detail": "",
            }
        width = inference.get("input_width")
        height = inference.get("input_height")
        detail = f"{width}x{height}" if width and height else ""
        return {
            "id": "roi",
            "label": "ROI",
            "status": "ok",
            "message": "ROI 帧已生成",
            "detail": detail,
        }

    @staticmethod
    def _trace_inference_stage(inference: dict[str, Any]) -> dict[str, Any]:
        if not inference.get("ran"):
            return {
                "id": "inference",
                "label": "推理",
                "status": "blocked",
                "message": inference.get("reason") or "推理尚未执行",
                "detail": "",
            }
        if inference.get("available") is not True:
            return {
                "id": "inference",
                "label": "推理",
                "status": "failed",
                "message": inference.get("reason") or "推理失败",
                "detail": "",
            }
        mapped = int(inference.get("mapped_detections") or 0)
        if mapped <= 0:
            raw = int(inference.get("raw_detections") or 0)
            return {
                "id": "inference",
                "label": "推理",
                "status": "blocked",
                "message": "未检测到可用目标",
                "detail": f"raw={raw}, mapped={mapped}",
            }
        return {
            "id": "inference",
            "label": "推理",
            "status": "ok",
            "message": "推理有检测结果",
            "detail": f"mapped={mapped}",
        }

    def _trace_target_stage(
        self, context: FrameContext | None, control: dict[str, Any]
    ) -> dict[str, Any]:
        if context is None:
            return {
                "id": "target",
                "label": "目标",
                "status": "blocked",
                "message": "等待推理帧",
                "detail": "",
            }
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
            return {
                "id": "control",
                "label": "控制量",
                "status": "blocked",
                "message": "没有目标，未计算控制量",
                "detail": "",
            }
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
                "message": str(
                    control.get("no_send_reason")
                    or control.get("trigger_reason")
                    or "等待触发，控制量未发送"
                ),
                "detail": detail,
            }
        if round(dx) == 0 and round(dy) == 0:
            return {
                "id": "control",
                "label": "控制量",
                "status": "blocked",
                "message": str(control.get("reason") or "控制量为 0"),
                "detail": detail,
            }
        return {
            "id": "control",
            "label": "控制量",
            "status": "ok",
            "message": str(control.get("reason") or "控制量已生成"),
            "detail": detail,
        }

    @staticmethod
    def _trace_execution_stage(execution: dict[str, Any]) -> dict[str, Any]:
        if not execution:
            return {
                "id": "execution",
                "label": "执行",
                "status": "blocked",
                "message": "没有控制命令",
                "detail": "",
            }
        dx = float(execution.get("output_dx") or 0.0)
        dy = float(execution.get("output_dy") or 0.0)
        detail = f"{execution.get('executor_id') or ''} dx={dx:.0f}, dy={dy:.0f}"
        if execution.get("sent") is True:
            return {
                "id": "execution",
                "label": "执行",
                "status": "ok",
                "message": str(execution.get("message") or "已发送"),
                "detail": detail,
            }
        message = str(execution.get("message") or "未发送")
        status = (
            "failed"
            if "failed" in message.lower() or "unavailable" in message.lower()
            else "blocked"
        )
        return {
            "id": "execution",
            "label": "执行",
            "status": status,
            "message": message,
            "detail": detail,
        }

    def _record_control_frame(self) -> None:
        if not bool(getattr(getattr(self.config, "consumers", None), "recording", False)):
            return
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            return
        try:
            scheduler_status = self._scheduler_status()
            record = build_control_frame_record(
                control=self.last_control,
                target=self.last_target,
                inference=self.last_inference_status,
                execution=self.last_execution,
                config=self.config,
                scheduler_status=scheduler_status,
            )
            trace = build_control_trace_record(
                control=self.last_control,
                target=self.last_target,
                inference=self.last_inference_status,
                execution=self.last_execution,
                scheduler_status=scheduler_status,
            )
            trace_writer = getattr(recorder, "record_control_trace", None)
            writer = getattr(recorder, "record_control_frame", None)
            if callable(writer):
                writer(record)
            else:
                writer = getattr(recorder, "record", None)
                if callable(writer) and not callable(trace_writer):
                    writer(record)
            if callable(trace_writer):
                trace_writer(trace)
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
        payload = {
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
            "source_frame_id": getattr(intent, "source_frame_id", None),
            "source_track_id": getattr(intent, "source_track_id", None),
            "trajectory_generation": getattr(intent, "trajectory_generation", None),
            "intent": {
                "dx": float(getattr(intent, "dx", 0.0)),
                "dy": float(getattr(intent, "dy", 0.0)),
                "accepted": bool(getattr(intent, "accepted", False)),
                "clipped": bool(getattr(intent, "clipped", False)),
                "reason": str(getattr(intent, "reason", "")),
            },
        }
        if isinstance(metadata, dict):
            for key in (
                "device_send_start_ts_ns",
                "device_send_end_ts_ns",
                "device_send_clock_domain",
                "scheduler_send_delay_us",
            ):
                if key in metadata:
                    payload[key] = metadata[key]
        return payload

    def _attach_execution_to_last_control(self, execution: dict[str, Any]) -> None:
        if not isinstance(self.last_control, dict):
            return
        execution_frame_id = execution.get("source_frame_id")
        control_frame_id = self.last_control.get("frame_id")
        if (
            isinstance(execution_frame_id, int)
            and isinstance(control_frame_id, int)
            and execution_frame_id != control_frame_id
        ):
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
        pipeline = self.last_control.get("pipeline")
        if (
            isinstance(pipeline, dict)
            and str(pipeline.get("algorithm") or pipeline.get("algorithm_id"))
            == DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2
        ):
            pipeline["executor_success"] = bool(execution.get("sent", False))
            pipeline["executor_block_reason"] = str(metadata.get("block_reason") or "")
            pipeline["device_send_start_ts_ns"] = metadata.get("device_send_start_ts_ns")
            pipeline["device_send_end_ts_ns"] = metadata.get("device_send_end_ts_ns")

    def _source_width(self, frame: CapturedFrame) -> int:
        return int(frame.source_width or frame.width)

    def _source_height(self, frame: CapturedFrame) -> int:
        return int(frame.source_height or frame.height)

    @staticmethod
    def _capture_geometry(frame: CapturedFrame) -> tuple[int, int, str, bool]:
        source_width = getattr(frame, "source_width", None)
        source_height = getattr(frame, "source_height", None)
        if (
            isinstance(source_width, int)
            and isinstance(source_height, int)
            and source_width > 0
            and source_height > 0
        ):
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
            configured_name = str(profile[class_id]).strip()
            if configured_name:
                return configured_name
        if 0 <= class_id < len(context.classes):
            model_name = str(context.classes[class_id]).strip()
            if model_name:
                return model_name
        return f"未知类别（cls {class_id}）"

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
