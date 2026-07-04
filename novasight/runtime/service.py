from __future__ import annotations

from dataclasses import asdict
from typing import Any

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.control import PIDStrategy, PredictiveStrategy, ProportionalStrategy
from novasight.executors import ExecutorRegistry
from novasight.hardware import BoxInputState
from novasight.inference import InferenceResult
from novasight.model_registry import ModelRegistry
from novasight.contracts import ControlIntent, Detection, FrameContext, Track
from novasight.roi import center_roi_frame, map_detection_to_source

from .config_store import RuntimeConfigStore
from .state import RuntimeFrameResult, RuntimeState


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
        self.control_strategy = self._create_control_strategy(config)

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

    def process_frame(self, context: FrameContext) -> RuntimeFrameResult:
        self.last_frame_context = context
        intent = self._control_intent_from_context(context)
        control_intents = [intent] if intent is not None else []
        execution_results = [self.executors.execute(intent) for intent in control_intents]
        self.last_execution = (
            self._execution_result_payload(execution_results[-1])
            if execution_results
            else None
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
            roi_frame = center_roi_frame(frame, requested_size=self.config.roi.size)
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
                    map_detection_to_source(
                        Detection(
                            cls=item.cls,
                            score=item.score,
                            x=item.x,
                            y=item.y,
                            w=item.w,
                            h=item.h,
                        ),
                        offset_x=roi_frame.offset_x,
                        offset_y=roi_frame.offset_y,
                    )
                )
            detections = self._filter_detections_by_config(detections)
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
            width=self._source_width(frame),
            height=self._source_height(frame),
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
            "source_width": self._source_width(frame),
            "source_height": self._source_height(frame),
            "roi_offset_x": int(getattr(roi_frame, "offset_x", 0)),
            "roi_offset_y": int(getattr(roi_frame, "offset_y", 0)),
            "debug": dict(debug or {}),
        }

    def _control_intent_from_context(self, context: FrameContext) -> ControlIntent | None:
        target = self._select_control_target(context)
        if target is None:
            self.last_target = None
            self.last_control = None
            return None

        center = (context.width / 2, context.height / 2)
        box_input = self._box_input_state()
        command = self.control_strategy.calculate(target, center, box_input)
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
        can_emit = box_input.active or getattr(getattr(self.config, "hardware", None), "kind", "none") == "none"
        self.last_target = self._target_payload(target, context)
        self.last_control = {
            "frame_id": context.frame_id,
            "dx": command.dx,
            "dy": command.dy,
            "confidence": command.confidence,
            "reason": command.reason,
            "trigger_active": box_input.active,
            "will_emit": can_emit,
        }
        return intent if can_emit else None

    def _select_control_target(self, context: FrameContext) -> Track | Detection | None:
        candidates: list[Track | Detection] = list(context.tracks) or list(context.detections)
        if not candidates or context.width <= 0 or context.height <= 0:
            return None
        min_confidence = float(getattr(self.config.control, "min_confidence", 0.0))
        filtered = [item for item in candidates if float(item.score) >= min_confidence]
        if not filtered:
            return None
        center_x = context.width / 2
        center_y = context.height / 2
        radius = min(context.width, context.height) * float(self.config.control.fov_ratio)
        inside_fov = [
            item
            for item in filtered
            if ((item.cx - center_x) ** 2 + (item.cy - center_y) ** 2) ** 0.5 <= radius
        ]
        pool = inside_fov or filtered
        return max(
            pool,
            key=lambda item: (
                float(item.score),
                -((item.cx - center_x) ** 2 + (item.cy - center_y) ** 2),
            ),
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

    def _box_input_state(self) -> BoxInputState:
        if getattr(getattr(self.config, "hardware", None), "kind", "none") in {"", "none", "silent"}:
            return BoxInputState(left=True, raw={"mode": "diagnostic_auto_trigger"})
        button_reader = getattr(self.executors, "read_buttons", None)
        if callable(button_reader):
            try:
                buttons = button_reader()
            except Exception:
                buttons = {}
            if buttons.get("available") is True:
                return BoxInputState(
                    left=bool(buttons.get("left")),
                    right=bool(buttons.get("right")),
                    raw={"source": "kmnet_executor", **buttons},
                )
        getter = getattr(self.hardware, "get_input_state", None)
        if not callable(getter):
            return BoxInputState()
        try:
            state = getter()
        except Exception:
            return BoxInputState()
        return state if isinstance(state, BoxInputState) else BoxInputState()

    def _create_control_strategy(self, config: RuntimeConfig):
        strategy = getattr(config.control, "strategy", "pid")
        if strategy == "predictive":
            return PredictiveStrategy()
        if strategy == "proportional":
            return ProportionalStrategy(
                fov_ratio=config.control.fov_ratio,
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
        )

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
            "offset_x": float(target.cx - context.width / 2),
            "offset_y": float(target.cy - context.height / 2),
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
        }

    def _execution_result_payload(self, result: Any) -> dict[str, Any]:
        intent = getattr(result, "intent", None)
        return {
            "executor_id": str(getattr(result, "executor_id", "")),
            "sent": bool(getattr(result, "sent", False)),
            "message": str(getattr(result, "message", "")),
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
