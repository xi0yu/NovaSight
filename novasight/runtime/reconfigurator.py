from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from novasight.config import RuntimeConfig, parse_runtime_config, save_runtime_config
from novasight.config.schema import runtime_config_schema
from novasight.executors import ExecutorRegistry
from novasight.inference.jetson import create_gpu_resource_preprocessor
from novasight.runtime.pipeline_factory import create_runtime_pipeline


logger = logging.getLogger("novasight.runtime.reconfigurator")
PIPELINE_READY_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ConfigSectionApplyResult:
    section: str
    impact: str
    status: str
    message: str = ""


@dataclass(frozen=True)
class TargetingConfigPlan:
    aim_changed: bool
    profile_changed: bool
    filter_changed: bool
    priority_changed: bool
    labels_changed: bool

    @property
    def reset_control_history(self) -> bool:
        return self.aim_changed or self.profile_changed


@dataclass(frozen=True)
class ConfigApplyReport:
    config: dict[str, Any]
    schema: dict[str, Any] | None
    restart_required: bool
    applied: bool
    rolled_back: bool = False
    sections: list[ConfigSectionApplyResult] = field(default_factory=list)
    message: str = ""
    capture: dict[str, Any] | None = None

    def asdict(self, *, include_schema: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if self.schema is None or not include_schema:
            payload.pop("schema", None)
        payload["sections"] = [asdict(item) for item in self.sections]
        return payload


class RuntimeReconfigurator:
    """Applies runtime configuration with rollback around live module changes."""

    def __init__(self, app: Any) -> None:
        self.app = app
        lock = getattr(app.state, "runtime_reconfiguration_lock", None)
        if lock is None:
            lock = threading.RLock()
            app.state.runtime_reconfiguration_lock = lock
        self._lock = lock

    def apply(self, config: RuntimeConfig) -> ConfigApplyReport:
        with self._lock:
            return self._apply(config, include_schema=True)

    def apply_field(
        self,
        section: str,
        key: str,
        value: Any,
    ) -> ConfigApplyReport:
        """Atomically merge, validate, and apply one field against the latest config."""

        with self._lock:
            current = asdict(self.app.state.runtime.config_store.snapshot())
            section_value = current.get(str(section))
            if not isinstance(section_value, dict):
                raise ValueError(f"unknown runtime config section: {section}")
            section_value[str(key)] = value
            return self._apply(
                parse_runtime_config(current),
                include_schema=False,
            )

    def _apply(
        self,
        config: RuntimeConfig,
        *,
        include_schema: bool,
    ) -> ConfigApplyReport:
        previous_config = getattr(self.app.state, "config", None)
        targeting_plan = self._targeting_config_plan(previous_config, config)
        if targeting_plan is not None:
            return self._apply_targeting_config(
                config,
                previous_config,
                targeting_plan,
            )
        if self._power_saving_only_changed(previous_config, config):
            return self._apply_power_saving_config(config, previous_config)
        if (
            self._control_config_only_changed(previous_config, config)
            and not self._hardware_changed(previous_config, config)
        ):
            return self._apply_control_config(config, previous_config)
        previous_kmnet_status = self._kmnet_status()
        sections: list[ConfigSectionApplyResult] = []
        roi_changed = self._roi_changed(previous_config, config)
        pipeline_changed = self._pipeline_config_changed(previous_config, config)
        runtime = getattr(self.app.state, "runtime", None)
        pipeline = getattr(runtime, "pipeline", None) if runtime is not None else None
        was_running = bool(getattr(runtime, "running", False)) or bool(
            getattr(pipeline, "running", False)
        )

        if pipeline_changed:
            self._reset_runtime_pipeline("runtime pipeline source configuration changed")
            sections.append(
                ConfigSectionApplyResult(
                    section="runtime_pipeline",
                    impact="pipeline_rebuild",
                    status="stopped" if was_running else "cleared",
                    message="DeepStream pipeline-affecting config changed",
                )
            )
        try:
            self._install_config(config)
        except Exception as exc:
            if previous_config is not None:
                self._rollback(previous_config, previous_kmnet_status)
                if was_running:
                    self._ensure_runtime_pipeline_for_live_capture(required=False)
            raise ValueError(
                f"runtime config rejected; previous config restored: {exc}"
            ) from exc
        self._restore_live_executor_connection(previous_kmnet_status)
        sections.append(
            ConfigSectionApplyResult(
                section="runtime",
                impact="hot_config",
                status="applied",
                message="runtime config store and live executors updated",
            )
        )

        if roi_changed:
            try:
                self._reconfigure_live_capture_for_roi()
            except ValueError as exc:
                if previous_config is not None:
                    self._rollback(previous_config, previous_kmnet_status)
                sections.append(
                    ConfigSectionApplyResult(
                        section="roi",
                        impact="live_capture_rebuild",
                        status="rolled_back",
                        message=str(exc),
                    )
                )
                raise ValueError(str(exc)) from exc
            sections.append(
                ConfigSectionApplyResult(
                    section="roi",
                    impact="live_capture_rebuild",
                    status="applied",
                    message=f"center roi={config.roi.size}x{config.roi.size}",
                )
            )

        config_path = getattr(self.app.state, "config_path", None)
        if was_running:
            try:
                self._ensure_runtime_pipeline_for_live_capture(required=True)
            except ValueError as exc:
                if previous_config is not None:
                    self._rollback(previous_config, previous_kmnet_status)
                    if config_path is not None:
                        save_runtime_config(previous_config, config_path)
                    self._ensure_runtime_pipeline_for_live_capture(required=False)
                sections.append(
                    ConfigSectionApplyResult(
                        section="runtime_pipeline",
                        impact="pipeline_rebuild",
                        status="rolled_back",
                        message=str(exc),
                    )
                )
                raise ValueError(
                    f"runtime config rejected; previous config restored: {exc}"
                ) from exc
            if pipeline_changed:
                sections.append(
                    ConfigSectionApplyResult(
                        section="runtime_pipeline",
                        impact="pipeline_rebuild",
                        status="ready",
                        message="new DeepStream pipeline published its first valid DetectionBatch",
                    )
                )
        if config_path is not None:
            save_runtime_config(config, config_path)
        return ConfigApplyReport(
            config=asdict(config),
            schema=runtime_config_schema(config) if include_schema else None,
            restart_required=False,
            applied=True,
            sections=sections,
            message="配置已应用到运行态",
        )

    def _apply_targeting_config(
        self,
        config: RuntimeConfig,
        previous_config: RuntimeConfig,
        plan: TargetingConfigPlan,
    ) -> ConfigApplyReport:
        runtime = self.app.state.runtime
        config_path = getattr(self.app.state, "config_path", None)
        try:
            runtime.update_targeting_config(
                config,
                reset_control_history=plan.reset_control_history,
            )
            self.app.state.config = config
            if config_path is not None:
                save_runtime_config(config, config_path)
        except Exception as exc:
            runtime.update_targeting_config(
                previous_config,
                reset_control_history=plan.reset_control_history,
            )
            self.app.state.config = previous_config
            if config_path is not None:
                save_runtime_config(previous_config, config_path)
            raise ValueError(
                f"targeting config rejected; previous config restored: {exc}"
            ) from exc
        return ConfigApplyReport(
            config=asdict(config),
            schema=None,
            restart_required=False,
            applied=True,
            sections=[
                ConfigSectionApplyResult(
                    section="control.aim" if plan.aim_changed else "targeting",
                    impact=(
                        "aim_mapping_hot_update"
                        if plan.aim_changed
                        else "targeting_hot_update"
                    ),
                    status="applied",
                    message="targeting config updated without rebuilding runtime modules",
                )
            ],
            message="目标配置已热更新",
        )

    def _apply_control_config(
        self,
        config: RuntimeConfig,
        previous_config: RuntimeConfig,
    ) -> ConfigApplyReport:
        runtime = self.app.state.runtime
        executors = self.app.state.executors
        config_path = getattr(self.app.state, "config_path", None)
        try:
            runtime.update_config(
                config,
                executors=executors,
                reconfigure_inference=False,
            )
            self.app.state.config = config
            if config_path is not None:
                save_runtime_config(config, config_path)
        except Exception as exc:
            runtime.update_config(
                previous_config,
                executors=executors,
                reconfigure_inference=False,
            )
            self.app.state.config = previous_config
            if config_path is not None:
                save_runtime_config(previous_config, config_path)
            raise ValueError(
                f"control config rejected; previous config restored: {exc}"
            ) from exc
        return ConfigApplyReport(
            config=asdict(config),
            schema=None,
            restart_required=False,
            applied=True,
            sections=[
                ConfigSectionApplyResult(
                    section="control",
                    impact="control_hot_update",
                    status="applied",
                    message="control config updated without touching inference or pipeline",
                )
            ],
            message="控制配置已热更新",
        )

    def _apply_power_saving_config(
        self,
        config: RuntimeConfig,
        previous_config: RuntimeConfig,
    ) -> ConfigApplyReport:
        supervisor = getattr(self.app.state, "runtime_power", None)
        if supervisor is None or not callable(getattr(supervisor, "reconfigure", None)):
            raise ValueError("runtime power supervisor is unavailable")
        config_path = getattr(self.app.state, "config_path", None)
        policy = config.power_saving
        policy_transition_started = False
        try:
            if config_path is not None:
                save_runtime_config(config, config_path)
            policy_transition_started = True
            supervisor.reconfigure(
                enabled=policy.host_presence_enabled,
                target_host_id=policy.target_host_id,
                heartbeat_timeout_s=policy.heartbeat_timeout_s,
                offline_grace_s=policy.offline_grace_s,
                auto_resume=policy.auto_resume,
            )
            self.app.state.config = config
            self.app.state.runtime.config = config
            self.app.state.runtime.config_store.replace(config)
        except Exception as exc:
            previous = previous_config.power_saving
            if policy_transition_started:
                supervisor.reconfigure(
                    enabled=previous.host_presence_enabled,
                    target_host_id=previous.target_host_id,
                    heartbeat_timeout_s=previous.heartbeat_timeout_s,
                    offline_grace_s=previous.offline_grace_s,
                    auto_resume=previous.auto_resume,
                )
            self.app.state.config = previous_config
            self.app.state.runtime.config = previous_config
            self.app.state.runtime.config_store.replace(previous_config)
            if config_path is not None:
                save_runtime_config(previous_config, config_path)
            raise ValueError(f"power-saving config rejected; previous config restored: {exc}") from exc
        return ConfigApplyReport(
            config=asdict(config),
            schema=None,
            restart_required=False,
            applied=True,
            sections=[ConfigSectionApplyResult(
                section="power_saving",
                impact="policy_hot_update",
                status="applied",
                message="host-presence policy updated without rebuilding runtime pipeline",
            )],
            message="省流策略已热更新",
        )

    def select_capture(
        self,
        *,
        device: str,
        preference: str | None = None,
        pixel_format: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> ConfigApplyReport:
        with self._lock:
            return self._select_capture(
                device=device,
                preference=preference,
                pixel_format=pixel_format,
                width=width,
                height=height,
                fps=fps,
            )

    def _select_capture(
        self,
        *,
        device: str,
        preference: str | None = None,
        pixel_format: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> ConfigApplyReport:
        previous_config = getattr(self.app.state, "config", None)
        previous_capture_signature = self._capture_signature(previous_config)
        runtime = getattr(self.app.state, "runtime", None)
        pipeline = getattr(runtime, "pipeline", None) if runtime is not None else None
        was_running = bool(getattr(runtime, "running", False)) or bool(
            getattr(pipeline, "running", False)
        )
        backend = str(self.app.state.config.inference.backend).lower()
        state = self.app.state.capture.configure_profile_only(
            device,
            preference=preference,
            pixel_format=pixel_format,
            width=width,
            height=height,
            fps=fps,
        )
        capture_payload = asdict(state)
        config_error = getattr(self.app.state.capture, "last_config_error", None)
        if config_error is not None:
            return ConfigApplyReport(
                config=asdict(self.app.state.config),
                schema=runtime_config_schema(self.app.state.config),
                restart_required=bool(getattr(getattr(self.app.state, "runtime", None), "running", False)),
                applied=False,
                rolled_back=True,
                sections=[
                    ConfigSectionApplyResult(
                        section="capture",
                        impact="live_capture_rebuild",
                        status="rolled_back",
                        message=str(config_error.last_error),
                    )
                ],
                message="采集切换失败，上一组可用采集链路已保留",
                capture=asdict(config_error),
            )
        if state.available is False:
            return ConfigApplyReport(
                config=asdict(self.app.state.config),
                schema=runtime_config_schema(self.app.state.config),
                restart_required=bool(getattr(getattr(self.app.state, "runtime", None), "running", False)),
                applied=False,
                rolled_back=True,
                sections=[
                    ConfigSectionApplyResult(
                        section="capture",
                        impact="live_capture_rebuild",
                        status="failed",
                        message=str(state.last_error or "capture unavailable"),
                    )
                ],
                message="采集切换失败",
                capture=capture_payload,
            )
        config = self.app.state.config
        config.source.default = "capture"
        config.capture.device = state.profile.device if state.profile is not None else state.device
        config.capture.preference = "manual"
        if state.profile is not None:
            config.capture.pixel_format = state.profile.pixel_format
            config.capture.width = state.profile.width
            config.capture.height = state.profile.height
            config.capture.fps = state.profile.fps
        self.app.state.capture.roi_size = config.roi.size
        self.app.state.runtime.update_config(config)
        config_path = getattr(self.app.state, "config_path", None)
        if config_path is not None:
            save_runtime_config(config, config_path)
        sections = [
            ConfigSectionApplyResult(
                section="capture",
                impact="live_capture_rebuild",
                status="applied",
                message=(
                    f"{config.capture.pixel_format} {config.capture.width}x"
                    f"{config.capture.height}@{config.capture.fps}"
                ),
            )
        ]
        if previous_capture_signature != self._capture_signature(config):
            self._reset_runtime_pipeline("capture selection changed")
            sections.append(
                ConfigSectionApplyResult(
                    section="runtime_pipeline",
                    impact="pipeline_rebuild",
                    status="cleared",
                    message="capture selection changed; restart runtime to rebuild GStreamer pipeline",
                )
            )
        if was_running:
            try:
                self._ensure_runtime_pipeline_for_live_capture(required=True)
            except ValueError as exc:
                sections.append(
                    ConfigSectionApplyResult(
                        section="runtime_pipeline",
                        impact="pipeline_rebuild",
                        status="failed",
                        message=str(exc),
                    )
                )
                return ConfigApplyReport(
                    config=asdict(config),
                    schema=runtime_config_schema(config),
                    restart_required=False,
                    applied=False,
                    rolled_back=False,
                    sections=sections,
                    message=f"采集配置已应用，但启动主链失败：{exc}",
                    capture=capture_payload,
                )
        return ConfigApplyReport(
            config=asdict(config),
            schema=runtime_config_schema(config),
            restart_required=bool(getattr(self.app.state.runtime, "running", False)),
            applied=True,
            sections=sections,
            message="采集配置已应用",
            capture=capture_payload,
        )

    def _install_config(self, config: RuntimeConfig) -> None:
        previous_config = getattr(self.app.state, "config", None)
        hardware_changed = self._hardware_changed(previous_config, config)
        current_executors = getattr(self.app.state, "executors", None)
        if current_executors is not None and not hardware_changed:
            next_executors = current_executors
        else:
            if current_executors is not None:
                self._disconnect_executor_registry(current_executors)
            next_executors = ExecutorRegistry.from_config(config)
        self.app.state.config = config
        capture_session = getattr(self.app.state.capture, "session", None)
        if capture_session is not None and getattr(capture_session, "running", False):
            self.app.state.capture.stop(
                "capture ownership transferred to deepstream_nvinfer"
            )
        self.app.state.capture.config = config.capture
        self.app.state.capture.roi_size = config.roi.size
        self.app.state.executors = next_executors
        self.app.state.inference.configure(
            confidence_threshold=config.inference.confidence_threshold,
            nms_threshold=config.inference.nms_threshold,
            gpu_preprocessor=create_gpu_resource_preprocessor(config),
        )
        self.app.state.inference.unload(
            "TensorRT engine ownership delegated to DeepStream nvinfer"
        )
        self.app.state.runtime.update_config(
            config,
            executors=self.app.state.executors,
        )

    @staticmethod
    def _disconnect_executor_registry(executors: ExecutorRegistry) -> None:
        kmnet = executors.executors.get("kmnet")
        disconnect = getattr(kmnet, "disconnect", None)
        if not callable(disconnect):
            return
        try:
            disconnect()
        except Exception as exc:
            raise RuntimeError(
                f"cannot replace kmNet executor before releasing its monitor: {exc}"
            ) from exc

    def _rollback(
        self,
        previous_config: RuntimeConfig,
        previous_kmnet_status: dict[str, Any],
    ) -> None:
        self._install_config(previous_config)
        self._restore_live_executor_connection(previous_kmnet_status)

    def _kmnet_status(self) -> dict[str, Any]:
        previous_executors = getattr(self.app.state, "executors", None)
        previous_kmnet = getattr(previous_executors, "executors", {}).get("kmnet")
        if previous_kmnet is None or not callable(getattr(previous_kmnet, "status", None)):
            return {}
        return previous_kmnet.status()

    @staticmethod
    def _roi_changed(previous_config: RuntimeConfig | None, config: RuntimeConfig) -> bool:
        if previous_config is None:
            return False
        return previous_config.roi.size != config.roi.size

    @staticmethod
    def _hardware_changed(previous_config: RuntimeConfig | None, config: RuntimeConfig) -> bool:
        if previous_config is None:
            return True
        return (
            previous_config.hardware.host != config.hardware.host
            or previous_config.hardware.port != config.hardware.port
            or previous_config.hardware.uuid != config.hardware.uuid
            or previous_config.hardware.monitor_port != config.hardware.monitor_port
            or previous_config.control.trigger_mode != config.control.trigger_mode
            or previous_config.control.shared.recoil_enabled
            != config.control.shared.recoil_enabled
        )

    @staticmethod
    def _power_saving_only_changed(
        previous_config: RuntimeConfig | None,
        config: RuntimeConfig,
    ) -> bool:
        if previous_config is None or previous_config.power_saving == config.power_saving:
            return False
        previous = asdict(previous_config)
        current = asdict(config)
        previous.pop("power_saving", None)
        current.pop("power_saving", None)
        return previous == current

    @staticmethod
    def _capture_signature(config: RuntimeConfig | None) -> tuple[object, ...]:
        if config is None:
            return ()
        capture = config.capture
        return (
            capture.device,
            capture.backend,
            capture.memory,
            capture.pixel_format,
            capture.width,
            capture.height,
            capture.fps,
        )

    def _reconfigure_live_capture_for_roi(self) -> None:
        # ROI is owned by the rebuilt DeepStream pipeline; no secondary
        # appsink capture session may be started during reconfiguration.
        return

    def _restore_live_executor_connection(self, previous_kmnet_status: dict[str, Any]) -> None:
        if previous_kmnet_status.get("connected") is not True:
            return
        kmnet = self.app.state.executors.executors.get("kmnet")
        connect = getattr(kmnet, "connect", None)
        if not callable(connect):
            return
        try:
            status = connect()
        except Exception as exc:
            logger.warning("kmNet reconnect after config update failed: %s", exc)
            return
        if status.get("connected") is True:
            logger.info("kmNet connection restored after config update")
        else:
            logger.warning(
                "kmNet reconnect after config update did not connect: %s",
                status.get("last_error") or status,
            )

    def _ensure_runtime_pipeline_for_live_capture(self, *, required: bool = False) -> None:
        runtime = getattr(self.app.state, "runtime", None)
        capture = getattr(self.app.state, "capture", None)
        config = getattr(self.app.state, "config", None)
        if runtime is None or capture is None or config is None:
            return
        backend = str(getattr(getattr(config, "inference", None), "backend", "")).lower()
        if backend != "deepstream_nvinfer":
            return
        if not bool(getattr(getattr(config, "inference", None), "enabled", True)):
            return
        try:
            if runtime.pipeline is None:
                runtime.pipeline = create_runtime_pipeline(capture=capture, runtime=runtime)
            if getattr(runtime.pipeline, "running", False):
                return
            runtime.pipeline.start()
            wait_until_ready = getattr(runtime.pipeline, "wait_until_ready", None)
            if callable(wait_until_ready) and not bool(
                wait_until_ready(PIPELINE_READY_TIMEOUT_S)
            ):
                status = runtime.pipeline.status() if callable(
                    getattr(runtime.pipeline, "status", None)
                ) else {}
                deepstream_status = (
                    status.get("deepstream") if isinstance(status, dict) else None
                )
                detail = str(
                    (status.get("last_error") if isinstance(status, dict) else "")
                    or (
                        deepstream_status.get("last_error")
                        if isinstance(deepstream_status, dict)
                        else ""
                    )
                    or "first valid DetectionBatch timed out"
                )
                raise RuntimeError(
                    "DeepStream pipeline did not become ready within "
                    f"{PIPELINE_READY_TIMEOUT_S:.1f}s: {detail}"
                )
        except Exception as exc:
            self._reset_runtime_pipeline("runtime pipeline restart failed")
            if required:
                raise ValueError(str(exc)) from exc
            logger.warning("runtime pipeline restart after config update failed: %s", exc)
        else:
            logger.info("runtime pipeline restarted after config update")

    def _reset_runtime_pipeline(self, reason: str) -> None:
        runtime = getattr(self.app.state, "runtime", None)
        pipeline = getattr(runtime, "pipeline", None) if runtime is not None else None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                logger.warning("runtime pipeline stop during config update failed: %s", exc)
        if runtime is not None:
            runtime.pipeline = None
            runtime.running = False
        logger.info("runtime pipeline cleared after config update: %s", reason)

    @staticmethod
    def _pipeline_config_changed(
        previous_config: RuntimeConfig | None,
        config: RuntimeConfig,
    ) -> bool:
        if previous_config is None:
            return False
        previous_inference = previous_config.inference
        next_inference = config.inference
        previous_capture = previous_config.capture
        next_capture = config.capture
        previous_preprocess = previous_config.preprocess
        next_preprocess = config.preprocess
        previous_runtime = previous_config.runtime
        next_runtime = config.runtime
        previous_roi = previous_config.roi
        next_roi = config.roi
        previous_crosshair = previous_config.crosshair
        next_crosshair = config.crosshair
        previous_limits = previous_config.limits
        next_limits = config.limits
        previous_consumers = previous_config.consumers
        next_consumers = config.consumers
        return (
            previous_inference.backend != next_inference.backend
            or previous_inference.device != next_inference.device
            or previous_inference.require_gpu != next_inference.require_gpu
            or previous_inference.allow_cpu_fallback != next_inference.allow_cpu_fallback
            or previous_inference.deepstream_io_mode != next_inference.deepstream_io_mode
            or previous_inference.deepstream_batched_push_timeout_us
            != next_inference.deepstream_batched_push_timeout_us
            or previous_inference.deepstream_parser_library
            != next_inference.deepstream_parser_library
            or previous_capture.backend != next_capture.backend
            or previous_capture.device != next_capture.device
            or previous_capture.memory != next_capture.memory
            or previous_capture.latest_only != next_capture.latest_only
            or previous_capture.appsink_max_buffers != next_capture.appsink_max_buffers
            or previous_capture.queue_leaky != next_capture.queue_leaky
            or previous_capture.pixel_format != next_capture.pixel_format
            or previous_capture.width != next_capture.width
            or previous_capture.height != next_capture.height
            or previous_capture.fps != next_capture.fps
            or previous_preprocess.backend != next_preprocess.backend
            or previous_preprocess.output_dtype != next_preprocess.output_dtype
            or previous_runtime.freshness_threshold_ms != next_runtime.freshness_threshold_ms
            or previous_runtime.drop_stale_batches != next_runtime.drop_stale_batches
            or previous_runtime.consume_latest_only != next_runtime.consume_latest_only
            or previous_roi.size != next_roi.size
            or previous_crosshair.enabled != next_crosshair.enabled
            or previous_crosshair.search_size != next_crosshair.search_size
            or previous_crosshair.sample_hz != next_crosshair.sample_hz
            or previous_limits.stream_fps != next_limits.stream_fps
            or previous_consumers.preview != next_consumers.preview
        )

    @staticmethod
    def _targeting_config_plan(
        previous_config: RuntimeConfig | None,
        config: RuntimeConfig,
    ) -> TargetingConfigPlan | None:
        if previous_config is None:
            return None
        previous_inference = previous_config.inference
        next_inference = config.inference
        plan = TargetingConfigPlan(
            aim_changed=previous_config.control.aim != config.control.aim,
            profile_changed=(
                previous_inference.detection_class_profile
                != next_inference.detection_class_profile
            ),
            filter_changed=(
                previous_inference.detection_class_filter
                != next_inference.detection_class_filter
            ),
            priority_changed=(
                previous_inference.detection_class_priority
                != next_inference.detection_class_priority
            ),
            labels_changed=(
                previous_inference.detection_class_profiles
                != next_inference.detection_class_profiles
            ),
        )
        if not any(
            (
                plan.aim_changed,
                plan.profile_changed,
                plan.filter_changed,
                plan.priority_changed,
                plan.labels_changed,
            )
        ):
            return None
        previous_payload = asdict(previous_config)
        next_payload = asdict(config)
        previous_payload["control"]["aim"] = next_payload["control"]["aim"]
        for key in (
            "detection_class_profile",
            "detection_class_filter",
            "detection_class_priority",
            "detection_class_profiles",
        ):
            previous_payload["inference"][key] = next_payload["inference"][key]
        return plan if previous_payload == next_payload else None

    @staticmethod
    def _control_config_only_changed(
        previous_config: RuntimeConfig | None,
        config: RuntimeConfig,
    ) -> bool:
        if previous_config is None or previous_config.control == config.control:
            return False
        previous_payload = asdict(previous_config)
        next_payload = asdict(config)
        previous_payload["control"] = next_payload["control"]
        return previous_payload == next_payload
