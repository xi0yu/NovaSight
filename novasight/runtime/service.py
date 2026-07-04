from __future__ import annotations

from dataclasses import asdict
import logging
import time
from typing import Any

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.control import PIDStrategy, PredictiveStrategy, ProportionalStrategy, StraightStrategy, aim_point
from novasight.executors import ExecutorRegistry
from novasight.hardware import BoxInputState
from novasight.inference import InferenceResult
from novasight.model_registry import ModelRegistry
from novasight.contracts import ControlIntent, Detection, FrameContext, Track
from novasight.roi import center_roi_frame

from .config_store import RuntimeConfigStore
from .state import RuntimeFrameResult, RuntimeState
from .target_selector import RuntimeTargetSelector, TargetSelection

logger = logging.getLogger("novasight.runtime.service")


class RuntimeService:
    def __init__(
        self,
        config: RuntimeConfig,
        models: ModelRegistry,
        executors: ExecutorRegistry,
        hardware: Any | None = None,
        capture: Any | None = None,
        inference: Any | None = None,
    ) -> None:
        self.config = config
        self.models = models
        self.executors = executors
        self.hardware = hardware
        self.capture = capture
        self.inference = inference
        self.running = False
        self.config_store = RuntimeConfigStore(config)
        self.pipeline = None
        self.fatal_error: dict | None = None
        self.last_frame_context: FrameContext | None = None
        self.last_target: dict[str, Any] | None = None
        self.last_control: dict[str, Any] | None = None
        self.last_execution: dict[str, Any] | None = None
        self.last_inference_reason = ""
        self.last_inference_status: dict[str, Any] = {
            "ran": False,
            "available": False,
            "reason": "推理尚未执行",
        }
        self._local_trigger_active = False
        self._local_trigger_bindings: list[str] = []
        self._local_trigger_updated_s = 0.0
        self._last_control_log_signature = ""
        self._last_control_log_s = 0.0
        self._last_trigger_log_signature = ""
        self._last_trigger_log_s = 0.0
        self._last_box_input_log_signature = ""
        self._last_box_input_log_s = 0.0
        self.control_strategy = self._create_control_strategy(config)
        self.target_selector = RuntimeTargetSelector()

    def state(self) -> RuntimeState:
        capture_state = getattr(self, "capture", None)
        inference_state = getattr(self, "inference", None)
        capture_payload = asdict(capture_state.state) if capture_state is not None else {}
        statistics = dict(capture_payload.get("statistics", {}))
        pipeline_stats = getattr(getattr(self, "pipeline", None), "stats", None)
        if pipeline_stats is not None:
            statistics["inference_counter"] = getattr(pipeline_stats, "window_processed_frames", 0)
            statistics["skipped_counter"] = getattr(pipeline_stats, "skipped_frames", 0)
            statistics["inference_fps"] = getattr(pipeline_stats, "inference_fps", 0.0)
            statistics["e2e_latency"] = getattr(pipeline_stats, "e2e_latency_ms", 0.0)
        if capture_payload:
            capture_payload["statistics"] = statistics
        return RuntimeState(
            running=self.running,
            source=self.config.source.default,
            active_model=self._active_model(),
            executor=self.executors.status(),
            capture=capture_payload,
            statistics=statistics,
            inference=(
                inference_state.status()
                if inference_state is not None
                else {"available": False}
            ),
            config=self.config_store.status(),
            pipeline=self.pipeline.status() if self.pipeline is not None else {},
            vision=self._vision_status(),
            fatal_error=self.fatal_error,
        )

    def update_config(self, config: RuntimeConfig) -> RuntimeConfig:
        self.config = config
        self.control_strategy = self._create_control_strategy(config)
        self.target_selector.reset()
        configure = getattr(self.inference, "configure", None)
        if callable(configure):
            configure(
                confidence_threshold=config.inference.confidence_threshold,
                nms_threshold=config.inference.nms_threshold,
            )
        return self.config_store.replace(config)

    def record_fatal_error(self, thread_name: str, exc: BaseException, path) -> None:
        self.fatal_error = {
            "type": "FATAL_ERROR",
            "thread": thread_name,
            "message": str(exc),
            "crash_log": str(path),
        }

    def update_local_trigger(self, *, active: bool, bindings: list[str] | None = None) -> dict[str, Any]:
        configured = self._configured_trigger_bindings()
        normalized = self._normalize_trigger_bindings(bindings if bindings is not None else configured)
        allowed = {item.lower() for item in configured}
        accepted = bool(active) and bool(normalized) and any(item.lower() in allowed for item in normalized)
        self._local_trigger_active = accepted
        self._local_trigger_bindings = normalized
        self._local_trigger_updated_s = time.monotonic()
        self._log_local_trigger_update(
            active=bool(active),
            accepted=accepted,
            configured=configured,
            pressed=normalized,
        )
        return {
            "available": bool(configured),
            "active": self._local_trigger_active,
            "bindings": list(configured),
            "pressed": list(normalized),
            "updated_ms": int(self._local_trigger_updated_s * 1000),
        }

    def process_frame(self, context: FrameContext) -> RuntimeFrameResult:
        self.last_frame_context = context
        intent = self._control_intent_from_context(context)
        control_intents = [intent] if intent is not None else []
        execution_results = [self.executors.execute(intent) for intent in control_intents]
        if execution_results:
            self.last_execution = self._execution_result_payload(execution_results[-1])
            logger.info(
                "control execution result frame=%s executor=%s sent=%s dx=%.1f dy=%.1f message=%s",
                context.frame_id,
                self.last_execution.get("executor_id"),
                self.last_execution.get("sent"),
                float(self.last_execution.get("output_dx") or 0.0),
                float(self.last_execution.get("output_dy") or 0.0),
                self.last_execution.get("message"),
            )
        return RuntimeFrameResult(
            control_intents=control_intents,
            execution_results=execution_results,
        )

    def process_captured_frame(self, frame: CapturedFrame) -> RuntimeFrameResult:
        if self.inference is None:
            self._record_inference_status(
                frame=frame,
                ran=False,
                available=False,
                reason="推理运行时未初始化",
            )
            return self.process_frame(self._empty_frame_context(frame))

        infer = getattr(self.inference, "infer", None)
        if not callable(infer):
            self._record_inference_status(
                frame=frame,
                ran=False,
                available=False,
                reason="推理运行时没有 infer 方法",
            )
            return self.process_frame(self._empty_frame_context(frame))

        try:
            roi_frame = center_roi_frame(
                frame,
                requested_size=self.config.roi.size,
                offset_x=self.config.roi.offset_x,
                offset_y=self.config.roi.offset_y,
            )
            inference_result = infer(roi_frame)
        except Exception as exc:
            self.last_inference_reason = str(exc)
            self._record_inference_status(
                frame=frame,
                roi_frame=locals().get("roi_frame"),
                ran=True,
                available=False,
                reason=str(exc),
            )
            return self.process_frame(self._empty_frame_context(frame))

        if not isinstance(inference_result, InferenceResult):
            self.last_inference_reason = "invalid inference result"
            self._record_inference_status(
                frame=frame,
                roi_frame=roi_frame,
                ran=True,
                available=False,
                reason=self.last_inference_reason,
            )
            return self.process_frame(self._empty_frame_context(frame))

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
            return self.process_frame(self._empty_frame_context(frame))

        try:
            detections = []
            for item in inference_result.detections:
                detections.append(
                    Detection(
                        cls=item.cls,
                        score=item.score,
                        x=item.x,
                        y=item.y,
                        w=item.w,
                        h=item.h,
                    )
                )
            classes = list(inference_result.classes)
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
            return self.process_frame(self._empty_frame_context(frame))
        self.last_inference_reason = ""
        self._record_inference_status(
            frame=frame,
            roi_frame=roi_frame,
            ran=True,
            available=True,
            reason="",
            raw_detections=len(inference_result.detections),
            mapped_detections=len(detections),
            classes=classes,
            debug=inference_result.debug,
        )

        context = FrameContext(
            frame_id=frame.frame_id,
            width=roi_frame.width,
            height=roi_frame.height,
            detections=detections,
            classes=classes,
        )
        return self.process_frame(context)

    def _empty_frame_context(self, frame: CapturedFrame) -> FrameContext:
        return FrameContext(
            frame_id=frame.frame_id,
            width=self._source_width(frame),
            height=self._source_height(frame),
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
        self.last_inference_status = {
            "frame_id": frame.frame_id,
            "ran": ran,
            "available": available,
            "reason": reason,
            "raw_detections": raw_detections,
            "mapped_detections": mapped_detections,
            "classes": list(classes or []),
            "input_width": int(getattr(roi_frame, "width", frame.width)),
            "input_height": int(getattr(roi_frame, "height", frame.height)),
            "input_image_width": self._image_width(getattr(roi_frame, "image", frame.image)),
            "input_image_height": self._image_height(getattr(roi_frame, "image", frame.image)),
            "input_pixel_format": str(getattr(roi_frame, "pixel_format", frame.pixel_format)),
            "model_input_width": model_input_width,
            "model_input_height": model_input_height,
            "model_to_roi_scale_x": model_to_roi_scale_x,
            "model_to_roi_scale_y": model_to_roi_scale_y,
            "input_downscale_factor": input_downscale_factor,
            "input_pixel_ratio": input_pixel_ratio,
            "input_density_warning": bool(preprocess_debug.get("density_warning", False)),
            "detection_coordinate_space": "roi",
            "source_width": self._source_width(frame),
            "source_height": self._source_height(frame),
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
        selection = self._select_control_target(context)
        target = selection.target
        if target is None:
            self._log_no_control_target(context, selection)
            self.last_target = None
            self.last_control = {
                "frame_id": context.frame_id,
                "selector_state": selection.state,
                "selection_reason": selection.reason,
                "candidates": selection.candidates,
                "inside_fov": selection.inside_fov,
                "lost_count": selection.lost_count,
                "will_emit": False,
            }
            self.last_execution = None
            return None

        center = (context.width / 2, context.height / 2)
        box_input = self._box_input_state()
        strategy_input = (
            box_input
            if box_input.active
            else BoxInputState(left=True, raw={"mode": "telemetry_control_calculation"})
        )
        aim_ratio = max(0.0, min(100.0, float(getattr(self.config.control, "aim_ratio", 40.0))))
        aim_x, aim_y = aim_point(target, aim_ratio)
        command = self.control_strategy.calculate(target, center, strategy_input)
        if not box_input.active:
            self._reset_control_motion_state()
        pipeline_debug = dict(command.debug)
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
        )
        output_mode = str(
            getattr(self.config.control, "output_mode", "")
            or getattr(getattr(self.config, "executor", None), "default", "")
        )
        hardware_kind = str(getattr(getattr(self.config, "hardware", None), "kind", "none"))
        trigger_mode = str(getattr(self.config.control, "trigger_mode", "hardware") or "hardware")
        if trigger_mode not in {"hardware", "telemetry", "always"}:
            trigger_mode = "hardware"
        requires_trigger = (
            output_mode == "kmnet"
            and hardware_kind not in {"", "none", "silent"}
            and trigger_mode != "always"
        )
        can_emit = box_input.active or not requires_trigger
        trigger_raw = getattr(box_input, "raw", {}) or {}
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
            "inside_fov": selection.inside_fov,
            "candidates": selection.candidates,
        }
        self.last_control = {
            "frame_id": context.frame_id,
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
            "selector_state": selection.state,
            "selection_reason": selection.reason,
            "target_detection_index": target_detection_index,
            "priority_rank": selection.priority_rank,
            "distance_px": selection.distance_px,
            "trigger_active": box_input.active,
            "trigger_required": requires_trigger,
            "trigger_mode": trigger_mode,
            "trigger_reason": str(trigger_raw.get("reason") or trigger_raw.get("mode") or trigger_raw.get("source") or ""),
            "output_mode": output_mode,
            "will_emit": can_emit,
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
            self.last_execution = {
                "executor_id": str(getattr(self.executors, "selected", "")),
                "sent": False,
                "accepted": False,
                "clipped": False,
                "output_dx": 0.0,
                "output_dy": 0.0,
                "message": "等待硬件触发，控制量未发送",
                "intent": {
                    "dx": float(command.dx),
                    "dy": float(command.dy),
                    "accepted": False,
                    "clipped": False,
                    "reason": command.reason,
                },
            }
        return intent if can_emit else None

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
            "control decision frame=%s target_cls=%s score=%.3f dx=%.1f dy=%.1f emit=%s output=%s hardware=%s trigger=%s trigger_source=%s trigger_raw=%s reason=%s",
            context.frame_id,
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
        signature = (
            f"no_target|state={selection.state}|reason={selection.reason}|"
            f"dets={len(context.detections)}|tracks={len(context.tracks)}|"
            f"candidates={selection.candidates}|inside={selection.inside_fov}"
        )
        if not self._should_log("_last_control_log_signature", "_last_control_log_s", signature, interval_s=1.0):
            return
        logger.info(
            "control trace frame=%s stage=target state=%s reason=%s detections=%s tracks=%s candidates=%s inside_fov=%s class_filter=%s min_conf=%.3f fov_ratio=%.3f",
            context.frame_id,
            selection.state,
            selection.reason,
            len(context.detections),
            len(context.tracks),
            selection.candidates,
            selection.inside_fov,
            str(getattr(self.config.inference, "detection_class_filter", "all")),
            float(getattr(self.config.control, "min_confidence", 0.0)),
            float(getattr(self.config.control, "fov_ratio", 0.28)),
        )

    def _log_local_trigger_update(
        self,
        *,
        active: bool,
        accepted: bool,
        configured: list[str],
        pressed: list[str],
    ) -> None:
        signature = f"local|active={active}|accepted={accepted}|configured={configured}|pressed={pressed}"
        if not self._should_log("_last_trigger_log_signature", "_last_trigger_log_s", signature, interval_s=1.0):
            return
        logger.info(
            "trigger update source=local active=%s accepted=%s configured=%s pressed=%s",
            active,
            accepted,
            configured,
            pressed,
        )

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
            class_filter=str(getattr(self.config.inference, "detection_class_filter", "all")),
            class_priority=self._class_priority(),
            sticky_bias=float(getattr(self.config.control, "target_sticky_bias", 0.25)),
            lock_enabled=bool(getattr(self.config.control, "target_lock_enabled", True)),
            lost_grace_frames=int(getattr(self.config.control, "target_lost_grace_frames", 5)),
        )

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
        local = self._local_trigger_state()
        if getattr(getattr(self.config, "hardware", None), "kind", "none") in {"", "none", "silent"}:
            if local.active:
                self._log_box_input_state(local, "local_no_hardware")
                return local
            state = BoxInputState(left=True, raw={"mode": "diagnostic_auto_trigger"})
            self._log_box_input_state(state, "diagnostic_auto_trigger")
            return state
        button_reader = getattr(self.executors, "read_buttons", None)
        if callable(button_reader):
            try:
                buttons = button_reader()
            except Exception:
                buttons = {}
            if buttons.get("available") is True:
                hardware_state = BoxInputState(
                    left=bool(buttons.get("left")),
                    right=bool(buttons.get("right")),
                    raw={"source": "kmnet_executor", **buttons},
                )
                if local.active:
                    merged = self._merge_box_input(hardware_state, local)
                    self._log_box_input_state(merged, "hardware_plus_local")
                    return merged
                self._log_box_input_state(hardware_state, "hardware")
                return hardware_state
        if local.active:
            self._log_box_input_state(local, "local_fallback")
            return local
        getter = getattr(self.hardware, "get_input_state", None)
        if not callable(getter):
            state = BoxInputState(raw={"source": "none", "reason": "no input reader"})
            self._log_box_input_state(state, "none")
            return state
        try:
            state = getter()
        except Exception:
            state = BoxInputState(raw={"source": "hardware_box", "reason": "read exception"})
            self._log_box_input_state(state, "hardware_box_exception")
            return state
        result = state if isinstance(state, BoxInputState) else BoxInputState(raw={"source": "hardware_box", "reason": "invalid state"})
        self._log_box_input_state(result, "hardware_box")
        return result

    def _local_trigger_state(self) -> BoxInputState:
        if not self._local_trigger_active:
            return BoxInputState(raw={"source": "local_trigger", "active": False})
        if time.monotonic() - self._local_trigger_updated_s > 0.75:
            self._local_trigger_active = False
            self._local_trigger_bindings = []
            return BoxInputState(raw={"source": "local_trigger", "active": False, "reason": "expired"})
        return BoxInputState(
            side=True,
            raw={
                "source": "local_trigger",
                "active": True,
                "bindings": list(self._local_trigger_bindings),
            },
        )

    @staticmethod
    def _merge_box_input(first: BoxInputState, second: BoxInputState) -> BoxInputState:
        return BoxInputState(
            left=first.left or second.left,
            right=first.right or second.right,
            side=first.side or second.side,
            raw={
                "source": "merged",
                "hardware": first.raw,
                "local": second.raw,
            },
        )

    def _configured_trigger_bindings(self) -> list[str]:
        return self._normalize_trigger_bindings(
            list(getattr(getattr(self.config, "control", None), "trigger_bindings", []) or [])
        )

    @staticmethod
    def _normalize_trigger_bindings(bindings: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in bindings:
            normalized = str(item).strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            result.append(normalized)
            if len(result) >= 2:
                break
        return result

    def _create_control_strategy(self, config: RuntimeConfig):
        strategy = getattr(config.control, "strategy", "pid")
        if strategy == "straight":
            return StraightStrategy(
                aim_y_ratio=config.control.aim_ratio,
                fov_deg=config.control.straight_fov_deg,
                counts_per_revolution=config.control.straight_c360,
                kp_x=config.control.straight_kp_x,
                kp_y=config.control.straight_kp_y,
                first_frame_gain=config.control.straight_first_frame_gain,
                first_frame_max_step=config.control.straight_first_max_step,
                refine_max_step=config.control.straight_max_step,
                in_deadzone_px=config.control.straight_in_deadzone,
                jump_threshold_px=config.control.straight_jump_threshold,
                pred_gain=config.control.straight_pred_gain,
                pred_consistency_frames=config.control.straight_pred_consistency_frames,
                gain_y=config.control.straight_gain_y,
                move_kind=config.control.move_kind,
                move_ms=config.control.move_ms,
                trace_ms=config.control.trace_ms,
                bezier_curvature=config.control.bezier_curvature,
            )
        if strategy == "predictive":
            return PredictiveStrategy(aim_ratio=config.control.aim_ratio)
        if strategy == "proportional":
            return ProportionalStrategy(
                fov_ratio=config.control.fov_ratio,
                aim_ratio=config.control.aim_ratio,
                near_px=config.control.near_px,
                near_speed=config.control.near_speed,
                far_speed=config.control.far_speed,
                ema_alpha=config.control.ema_alpha,
                deadzone_counts=config.control.deadzone_counts,
                counts_per_revolution_x=config.control.counts_per_revolution_x,
                counts_per_revolution_y=config.control.counts_per_revolution_y,
                move_kind=config.control.move_kind,
                move_ms=config.control.move_ms,
                trace_ms=config.control.trace_ms,
                bezier_curvature=config.control.bezier_curvature,
            )
        return PIDStrategy(
            kp_x=config.control.pid_kp_x,
            kp_y=config.control.pid_kp_y,
            ki=config.control.pid_ki,
            kd=config.control.pid_kd,
            integral_limit=config.control.pid_integral_limit,
            move_limit=config.control.pid_move_limit,
            move_limit_x=config.control.kp_x_move_max,
            move_limit_y=config.control.kp_y_move_max,
            prediction_factor=config.control.prediction_factor,
            aim_ratio=config.control.aim_ratio,
            near_px=config.control.near_px,
            near_speed=config.control.near_speed,
            far_speed=config.control.far_speed,
            deadzone_counts=config.control.deadzone_counts,
            counts_per_revolution_x=config.control.counts_per_revolution_x,
            counts_per_revolution_y=config.control.counts_per_revolution_y,
            move_kind=config.control.move_kind,
            move_ms=config.control.move_ms,
            trace_ms=config.control.trace_ms,
            bezier_curvature=config.control.bezier_curvature,
        )

    def _reset_control_motion_state(self) -> None:
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

    def _target_payload(self, target: Track | Detection, context: FrameContext) -> dict[str, Any]:
        class_name = self._class_display_name(int(target.cls), context)
        payload = {
            "frame_id": context.frame_id,
            "class_id": int(target.cls),
            "class_name": class_name,
            "score": float(target.score),
            "x": float(target.x),
            "y": float(target.y),
            "w": float(target.w),
            "h": float(target.h),
            "cx": float(target.cx),
            "cy": float(target.cy),
            "box_cx": float(target.cx),
            "box_cy": float(target.cy),
            "offset_x": float(target.cx - context.width / 2),
            "offset_y": float(target.cy - context.height / 2),
            "box_offset_x": float(target.cx - context.width / 2),
            "box_offset_y": float(target.cy - context.height / 2),
        }
        if isinstance(target, Track):
            payload["track_id"] = int(target.track_id)
        return payload

    def _detection_payload(self, detection: Detection, context: FrameContext) -> dict[str, Any]:
        class_name = self._class_display_name(int(detection.cls), context)
        return {
            "frame_id": context.frame_id,
            "class_id": int(detection.cls),
            "class_name": class_name,
            "score": float(detection.score),
            "x": float(detection.x),
            "y": float(detection.y),
            "w": float(detection.w),
            "h": float(detection.h),
            "cx": float(detection.cx),
            "cy": float(detection.cy),
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
        detail = f"dx={dx:.1f}, dy={dy:.1f}"
        if control.get("will_emit") is not True:
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

    def _execution_result_payload(self, result: Any) -> dict[str, Any]:
        intent = getattr(result, "intent", None)
        return {
            "executor_id": str(getattr(result, "executor_id", "")),
            "sent": bool(getattr(result, "sent", False)),
            "message": str(getattr(result, "message", "")),
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

    def _source_width(self, frame: CapturedFrame) -> int:
        return int(frame.source_width or frame.width)

    def _source_height(self, frame: CapturedFrame) -> int:
        return int(frame.source_height or frame.height)

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
