from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from novasight.config import RuntimeConfig, save_runtime_config
from novasight.config.schema import runtime_config_schema
from novasight.executors import ExecutorRegistry
from novasight.hardware import create_hardware_box
from novasight.inference.jetson import create_gpu_resource_preprocessor
from novasight.runtime.pipeline import RuntimePipeline


logger = logging.getLogger("novasight.runtime.reconfigurator")


@dataclass(frozen=True)
class ConfigSectionApplyResult:
    section: str
    impact: str
    status: str
    message: str = ""


@dataclass(frozen=True)
class ConfigApplyReport:
    config: dict[str, Any]
    schema: dict[str, Any]
    restart_required: bool
    applied: bool
    rolled_back: bool = False
    sections: list[ConfigSectionApplyResult] = field(default_factory=list)
    message: str = ""
    capture: dict[str, Any] | None = None

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sections"] = [asdict(item) for item in self.sections]
        return payload


class RuntimeReconfigurator:
    """Applies runtime configuration with rollback around live module changes."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def apply(self, config: RuntimeConfig) -> ConfigApplyReport:
        previous_config = getattr(self.app.state, "config", None)
        previous_kmnet_status = self._kmnet_status()
        sections: list[ConfigSectionApplyResult] = []
        roi_changed = self._roi_changed(previous_config, config)
        pipeline_changed = self._pipeline_config_changed(previous_config, config)
        was_running = bool(getattr(getattr(self.app.state, "runtime", None), "running", False))

        self._install_config(config)
        if pipeline_changed:
            self._reset_runtime_pipeline("runtime pipeline source configuration changed")
            sections.append(
                ConfigSectionApplyResult(
                    section="runtime_pipeline",
                    impact="pipeline_rebuild",
                    status="stopped" if was_running else "cleared",
                    message="backend/capture/roi/model pipeline config changed",
                )
            )
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
                    message=f"roi={config.roi.size} offset=({config.roi.offset_x},{config.roi.offset_y})",
                )
            )

        config_path = getattr(self.app.state, "config_path", None)
        if config_path is not None:
            save_runtime_config(config, config_path)
        if was_running:
            self._ensure_runtime_pipeline_for_live_capture(required=True)
        running = bool(getattr(getattr(self.app.state, "runtime", None), "running", False))
        return ConfigApplyReport(
            config=asdict(config),
            schema=runtime_config_schema(config),
            restart_required=running,
            applied=True,
            sections=sections,
            message="配置已应用到运行态",
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
        previous_config = getattr(self.app.state, "config", None)
        previous_capture_signature = self._capture_signature(previous_config)
        runtime = getattr(self.app.state, "runtime", None)
        pipeline = getattr(runtime, "pipeline", None) if runtime is not None else None
        was_running = bool(getattr(runtime, "running", False)) or bool(
            getattr(pipeline, "running", False)
        )
        state = self.app.state.capture.configure(
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
        self.app.state.capture.roi_offset_x = config.roi.offset_x
        self.app.state.capture.roi_offset_y = config.roi.offset_y
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
            current_executors.update_runtime_config(config)
            next_executors = current_executors
            next_hardware = getattr(self.app.state, "hardware", None)
        else:
            next_executors = ExecutorRegistry.from_config(config)
            next_hardware = create_hardware_box(config)
        self.app.state.config = config
        self.app.state.capture.config = config.capture
        self.app.state.capture.roi_size = config.roi.size
        self.app.state.capture.roi_offset_x = config.roi.offset_x
        self.app.state.capture.roi_offset_y = config.roi.offset_y
        self.app.state.executors = next_executors
        self.app.state.hardware = next_hardware
        self.app.state.inference.configure(
            confidence_threshold=config.inference.confidence_threshold,
            nms_threshold=config.inference.nms_threshold,
            gpu_preprocessor=create_gpu_resource_preprocessor(config),
        )
        self.app.state.runtime.executors = self.app.state.executors
        self.app.state.runtime.hardware = self.app.state.hardware
        self.app.state.runtime.update_config(config)

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
        return (
            previous_config.roi.size != config.roi.size
            or previous_config.roi.offset_x != config.roi.offset_x
            or previous_config.roi.offset_y != config.roi.offset_y
        )

    @staticmethod
    def _hardware_changed(previous_config: RuntimeConfig | None, config: RuntimeConfig) -> bool:
        if previous_config is None:
            return True
        return (
            previous_config.hardware.kind != config.hardware.kind
            or previous_config.hardware.host != config.hardware.host
            or previous_config.hardware.port != config.hardware.port
            or previous_config.hardware.uuid != config.hardware.uuid
            or previous_config.hardware.monitor_port != config.hardware.monitor_port
        )

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
        capture = self.app.state.capture
        session = getattr(capture, "session", None)
        state = getattr(capture, "state", None)
        profile = getattr(state, "profile", None)
        if (
            profile is None
            or getattr(state, "available", False) is not True
            or (session is not None and getattr(session, "running", False) is not True)
        ):
            return
        if getattr(profile, "preference", "") == "image":
            return
        config = self.app.state.config
        logger.info(
            "capture roi changed; rebuilding live capture pipeline device=%s roi_size=%s offset=(%s,%s)",
            profile.device,
            config.roi.size,
            config.roi.offset_x,
            config.roi.offset_y,
        )
        new_state = capture.configure(
            profile.device,
            preference="manual",
            pixel_format=profile.pixel_format,
            width=profile.width,
            height=profile.height,
            fps=profile.fps,
        )
        config_error = getattr(capture, "last_config_error", None)
        if config_error is not None:
            raise ValueError(
                f"runtime config rejected; previous capture pipeline was restored: {config_error.last_error}"
            )
        if getattr(new_state, "available", False) is not True:
            raise ValueError(
                f"runtime config rejected; previous capture pipeline was restored: {new_state.last_error}"
            )

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
        if backend not in {"tensorrt", "nvmm_latest"}:
            return
        if not bool(getattr(getattr(config, "inference", None), "enabled", True)):
            return
        state = getattr(capture, "state", None)
        session = getattr(capture, "session", None)
        if (
            getattr(capture, "source", None) is None
            or getattr(state, "available", False) is not True
            or (session is not None and getattr(session, "running", False) is not True)
        ):
            return
        if runtime.pipeline is None:
            runtime.pipeline = RuntimePipeline(capture=capture, runtime=runtime)
        if getattr(runtime.pipeline, "running", False):
            return
        try:
            runtime.pipeline.start()
        except RuntimeError as exc:
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
        return (
            previous_inference.backend != next_inference.backend
            or previous_inference.device != next_inference.device
            or previous_inference.require_gpu != next_inference.require_gpu
            or previous_inference.allow_cpu_fallback != next_inference.allow_cpu_fallback
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
            or previous_roi.offset_x != next_roi.offset_x
            or previous_roi.offset_y != next_roi.offset_y
        )
