from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from novasight.model_ingress import InferenceConfigBuilder, ModelProfileStore
from novasight.roi import center_roi_region

from .backend import DeepStreamObjectBackend
from .nvinfer_config import write_nvinfer_config
from .pipeline_builder import DeepStreamPipelineConfig
from .model_manifest import ensure_engine_manifest, remove_matching_legacy_manifest


logger = logging.getLogger("novasight.deepstream.runtime")


@dataclass(slots=True)
class DeepStreamPipelineStats:
    consumed_detection_batches: int = 0
    processed_frames: int = 0
    control_observations: int = 0
    last_frame_id: int = -1
    last_generation: int = -1
    last_error: str = ""
    started_at: float | None = None
    stopped_at: float | None = None
    threads: dict[str, bool] = field(default_factory=dict)


class DeepStreamRuntimePipeline:
    """Consumes latest object-meta DetectionBatch objects into the shared runtime."""

    def __init__(self, *, backend: DeepStreamObjectBackend, runtime: Any) -> None:
        self.backend = backend
        self.runtime = runtime
        self.stats = DeepStreamPipelineStats()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads) and self.backend.running

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.stats = DeepStreamPipelineStats(started_at=time.time())
        self.backend.start()
        self.runtime.running = True
        self._threads = [
            threading.Thread(
                target=self._detection_loop,
                name="novasight-deepstream-detection",
                daemon=True,
            ),
            threading.Thread(
                target=self._control_loop,
                name="novasight-continuous-control",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.runtime.running = False
        cancel_control = getattr(self.runtime, "cancel_control", None)
        if callable(cancel_control):
            cancel_control("RUNTIME_STOPPED")
        self.backend.stop()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self.stats.stopped_at = time.time()

    def status(self) -> dict[str, object]:
        self.stats.threads = {thread.name: thread.is_alive() for thread in self._threads}
        return {
            **asdict(self.stats),
            "running": self.running,
            "selected": "deepstream_nvinfer",
            "deepstream": self.backend.status(),
        }

    def _detection_loop(self) -> None:
        mailbox = self.backend.detection_batch_mailbox
        after_generation = -1
        while not self._stop.is_set():
            batch = mailbox.acquire_latest(after_generation=after_generation, timeout_s=0.1)
            if batch is None:
                if not self.backend.running:
                    status = self.backend.status()
                    self.stats.last_error = str(
                        status.get("last_error") or "DeepStream backend stopped"
                    )
                    self.runtime.running = False
                    self._stop.set()
                    self.backend.stop()
                    break
                continue
            after_generation = int(batch.generation or 0)
            self.stats.consumed_detection_batches += 1
            self.stats.last_generation = after_generation
            self.stats.last_frame_id = int(batch.frame_id)
            if self.stats.consumed_detection_batches == 1:
                logger.info(
                    "DeepStream runtime consumed first DetectionBatch frame=%s generation=%s "
                    "detections=%s age_ms=%.2f",
                    batch.frame_id,
                    batch.generation,
                    len(batch.detections),
                    float(batch.result_age_ms or 0.0),
                )
            try:
                result = self.runtime.process_detection_batch(
                    batch,
                    width=self.backend.pipeline_config.roi_width,
                    height=self.backend.pipeline_config.roi_height,
                    source_width=self.backend.pipeline_config.capture_width,
                    source_height=self.backend.pipeline_config.capture_height,
                    roi_offset_x=self.backend.pipeline_config.roi_left,
                    roi_offset_y=self.backend.pipeline_config.roi_top,
                )
            except Exception as exc:
                logger.exception("DeepStream DetectionBatch handoff failed")
                self.stats.last_error = str(exc)
                self.runtime.running = False
                self._stop.set()
                self.backend.stop()
                break
            self.stats.processed_frames += 1
            if getattr(result, "observation_updated", False) is True:
                self.stats.control_observations += 1

    def _control_loop(self) -> None:
        process_control_tick = getattr(self.runtime, "process_control_tick", None)
        if not callable(process_control_tick):
            return
        while not self._stop.is_set():
            executors = getattr(self.runtime, "executors", None)
            if getattr(executors, "scheduler", None) is None:
                time.sleep(0.05)
                continue
            process_control_tick()
            config = getattr(getattr(self.runtime, "config", None), "control", None)
            interval_ms = max(
                1.0,
                min(10.0, float(getattr(config, "scheduler_interval_ms", 4.0))),
            )
            time.sleep(interval_ms / 1000.0)


def create_deepstream_runtime_pipeline(*, runtime: Any) -> DeepStreamRuntimePipeline:
    config = runtime.config
    models = runtime.models
    if str(config.source.default).lower() != "capture":
        raise RuntimeError("DeepStream nvinfer runtime requires source.default=capture")
    if bool(getattr(config.consumers, "inference", True)) is not True:
        raise RuntimeError("DeepStream nvinfer runtime requires consumers.inference=true")
    deployment = models.get_active_deployment()
    if deployment is None:
        raise RuntimeError("DeepStream runtime requires an active model deployment")
    artifact = models.get_artifact(deployment.artifact_id)
    if artifact is None or artifact.kind != "engine":
        raise RuntimeError("DeepStream runtime requires an active TensorRT .engine artifact")
    version = models.get_version(artifact.version_id)
    project = models.get_project(version.project_id) if version is not None else None
    if version is None or project is None:
        raise RuntimeError("active DeepStream model registry references are incomplete")
    engine_path = models.resolve_artifact_path(artifact)
    profile_store = ModelProfileStore()
    profile_path = profile_store.existing_path_for_engine(engine_path)
    if profile_store.contains_model_profile(profile_path):
        profile = profile_store.load(profile_path)
        runtime_config = InferenceConfigBuilder().build(
            profile,
            parser_library_path=Path(config.inference.deepstream_parser_library),
        )
        manifest = runtime_config.manifest
        generated_manifest = False
    else:
        manifest, generated_manifest = ensure_engine_manifest(
            runtime.inference,
            engine_path=engine_path,
            model_id=project.name,
            display_name=project.name,
            classes=list(version.classes),
            registered_input_shape=version.input_shape,
            confidence_threshold=config.inference.confidence_threshold,
            nms_iou_threshold=config.inference.nms_threshold,
        )
    remove_matching_legacy_manifest(engine_path)
    resolved_classes = list(manifest.output.class_names)
    if list(version.classes) != resolved_classes:
        models.update_version_classes(version.id, resolved_classes)
        logger.warning(
            "synchronized model classes from DeepStream manifest path=%s classes=%s",
            engine_path,
            resolved_classes,
        )
    resolved_input_shape = "x".join(str(value) for value in manifest.input.shape)
    if version.input_shape != resolved_input_shape:
        models.update_version_input_shape(version.id, resolved_input_shape)
        logger.warning(
            "synchronized model input shape from TensorRT engine path=%s "
            "registered=%s engine=%s",
            engine_path,
            version.input_shape,
            resolved_input_shape,
        )
    if generated_manifest:
        models.update_artifact_status(
            artifact.id,
            "ready",
            checksum=manifest.artifact.sha256,
        )
        logger.info(
            "generated DeepStream model manifest from TensorRT engine path=%s fingerprint=%s",
            engine_path,
            manifest.model_fingerprint,
        )
    backend = _create_deepstream_backend(
        runtime=runtime,
        manifest=manifest,
        engine_path=engine_path,
        nvinfer_config_name="active-nvinfer.ini",
        preview_enabled=bool(getattr(config.consumers, "preview", True)),
    )
    return DeepStreamRuntimePipeline(backend=backend, runtime=runtime)


def create_deepstream_probe_backend(*, runtime: Any, profile: Any) -> DeepStreamObjectBackend:
    engine_path = Path(profile.engine.path).expanduser().resolve(strict=False)
    runtime_config = InferenceConfigBuilder().build(
        profile,
        parser_library_path=Path(runtime.config.inference.deepstream_parser_library),
    )
    fingerprint = str(runtime_config.manifest.model_fingerprint).replace("sha256:", "")[:16]
    return _create_deepstream_backend(
        runtime=runtime,
        manifest=runtime_config.manifest,
        engine_path=engine_path,
        nvinfer_config_name=f"probe-{fingerprint or 'candidate'}.ini",
        preview_enabled=False,
    )


def _create_deepstream_backend(
    *,
    runtime: Any,
    manifest: Any,
    engine_path: Path,
    nvinfer_config_name: str,
    preview_enabled: bool,
) -> DeepStreamObjectBackend:
    config = runtime.config
    models = runtime.models
    if len(manifest.input.shape) != 4 or str(manifest.input.layout).upper() != "NCHW":
        raise RuntimeError("DeepStream model input must be NCHW [N,C,H,W]")
    model_height = int(manifest.input.shape[2])
    model_width = int(manifest.input.shape[3])
    capture_width = int(config.capture.width)
    capture_height = int(config.capture.height)
    capture_fps = int(config.capture.fps)
    if capture_width <= 0 or capture_height <= 0 or capture_fps <= 0:
        raise RuntimeError(
            "DeepStream capture profile is incomplete; select width, height, and fps first"
        )
    roi_left, roi_top, roi_size = center_roi_region(
        source_width=capture_width,
        source_height=capture_height,
        requested_size=int(config.roi.size),
        offset_x=int(config.roi.offset_x),
        offset_y=int(config.roi.offset_y),
    )
    parser_library = Path(config.inference.deepstream_parser_library)
    runtime_dir = Path(models.data_dir).parent / "runtime" / "deepstream"
    nvinfer_config_path = write_nvinfer_config(
        runtime_dir / nvinfer_config_name,
        manifest,
        engine_path=engine_path,
        parser_library_path=parser_library,
        confidence_threshold=config.inference.confidence_threshold,
        nms_threshold=config.inference.nms_threshold,
    )
    pipeline_config = DeepStreamPipelineConfig(
        device=config.capture.device,
        capture_width=capture_width,
        capture_height=capture_height,
        fps=capture_fps,
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_size,
        roi_height=roi_size,
        model_width=model_width,
        model_height=model_height,
        nvinfer_config_path=nvinfer_config_path,
        pixel_format=config.capture.pixel_format,
        io_mode=config.inference.deepstream_io_mode,
        batched_push_timeout_us=config.inference.deepstream_batched_push_timeout_us,
        preview_enabled=preview_enabled,
        preview_fps=int(getattr(config.limits, "stream_fps", 30)),
    )
    return DeepStreamObjectBackend(
        pipeline_config=pipeline_config,
        manifest=manifest,
        parser_library_path=parser_library,
        max_publish_age_ms=_strictest_positive_ms(
            config.runtime.freshness_threshold_ms,
            config.inference.inference_input_deadline_ms,
        ),
    )


def _strictest_positive_ms(*values: object) -> float:
    parsed = [float(value) for value in values if float(value or 0.0) > 0.0]
    return min(parsed) if parsed else 0.0


__all__ = [
    "DeepStreamRuntimePipeline",
    "create_deepstream_probe_backend",
    "create_deepstream_runtime_pipeline",
]
