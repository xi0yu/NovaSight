import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import novasight.deepstream.backend as backend_module
import novasight.deepstream.runtime_pipeline as runtime_pipeline_module
from novasight.contracts import DetectionBatch
from novasight.config import RuntimeConfig
from novasight.deepstream.backend import DeepStreamDependencyStatus, DeepStreamObjectBackend
from novasight.deepstream.model_manifest import (
    PreparedEngineManifest,
    ensure_engine_manifest,
    remove_matching_legacy_manifest,
    resolve_yolo_class_contract,
)
from novasight.deepstream.nvinfer_config import generate_nvinfer_config
from novasight.deepstream.pipeline_builder import (
    DeepStreamPipelineConfig,
    build_deepstream_pipeline,
)
from novasight.model_registry.manifest import TensorSpec, build_engine_manifest, write_manifest
from novasight.model_registry import ModelRegistry, read_manifest
from novasight.deepstream.runtime_pipeline import (
    DeepStreamRuntimePipeline,
    create_deepstream_runtime_pipeline,
)
from novasight.runtime.detection_batch_mailbox import DetectionBatchMailbox


def _manifest(tmp_path: Path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    manifest = build_engine_manifest(
        model_id="target",
        display_name="Target",
        engine_path=engine,
        input_spec=TensorSpec("images", [1, 3, 320, 320], "float16", "NCHW"),
        output_spec=TensorSpec("output0", [1, 6, 2100], "float32", "NCHW"),
        class_count=2,
        class_names=["body", "head"],
        validated=True,
    )
    return engine, manifest


def _pipeline_config(tmp_path: Path) -> DeepStreamPipelineConfig:
    return DeepStreamPipelineConfig(
        device="/dev/video0",
        capture_width=1920,
        capture_height=1080,
        fps=120,
        roi_left=720,
        roi_top=300,
        roi_width=480,
        roi_height=480,
        model_width=320,
        model_height=320,
        nvinfer_config_path=tmp_path / "deepstream.ini",
    )


def test_deepstream_pipeline_is_nvmm_latest_only_with_hardware_jpeg_preview(tmp_path: Path) -> None:
    pipeline = build_deepstream_pipeline(_pipeline_config(tmp_path))

    assert pipeline.count("max-size-buffers=1") >= 4
    assert pipeline.count("leaky=downstream") >= 4
    assert "nvv4l2decoder mjpeg=1" in pipeline
    assert "nvvidconv left=720 right=1200 top=300 bottom=780" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=320,height=320" in pipeline
    assert "nvstreammux name=mux batch-size=1 live-source=1" in pipeline
    assert "nvinfer name=primary-infer" in pipeline
    assert "tee name=novasight_source_split" in pipeline
    assert "nvjpegenc name=preview-encoder" in pipeline
    assert "valve name=preview-valve drop=true" in pipeline
    assert pipeline.index("valve name=preview-valve") < pipeline.index("nvjpegenc name=preview-encoder")
    assert "appsink name=preview_sink" in pipeline
    assert "videoconvert" not in pipeline
    assert "video/x-raw,format=BGR" not in pipeline


def test_deepstream_pipeline_accepts_native_nv12_capture(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), pixel_format="NV12")

    pipeline = build_deepstream_pipeline(config)

    assert "video/x-raw,format=NV12,width=1920,height=1080,framerate=120/1" in pipeline
    assert "nvvidconv left=720 right=1200 top=300 bottom=780" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=480,height=480" in pipeline
    assert "jpegparse" not in pipeline
    assert "nvv4l2decoder" not in pipeline


def test_deepstream_pipeline_accepts_yuyv_capture_via_vic_conversion(tmp_path: Path) -> None:
    config = replace(_pipeline_config(tmp_path), pixel_format="YUYV")

    pipeline = build_deepstream_pipeline(config)

    assert "video/x-raw,format=YUY2,width=1920,height=1080,framerate=120/1" in pipeline
    assert "nvvidconv left=720 right=1200 top=300 bottom=780" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=480,height=480" in pipeline
    assert "jpegparse" not in pipeline
    assert "nvv4l2decoder" not in pipeline


def test_deepstream_pipeline_skips_model_resize_when_roi_already_matches(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    config = replace(
        config,
        roi_width=320,
        roi_height=320,
        preview_enabled=False,
    )

    pipeline = build_deepstream_pipeline(config)

    assert pipeline.count("nvvidconv") == 1


def test_deepstream_pipeline_fuses_crop_and_model_resize_without_side_branches(
    tmp_path: Path,
) -> None:
    config = replace(
        _pipeline_config(tmp_path),
        preview_enabled=False,
        crosshair_enabled=False,
    )

    pipeline = build_deepstream_pipeline(config)

    assert pipeline.count("nvvidconv") == 1
    assert "tee name=novasight_source_split" not in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=320,height=320" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=480,height=480" not in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=320,height=320" in pipeline


def test_nvinfer_config_uses_native_decode_and_exactly_one_nms(tmp_path: Path) -> None:
    engine, manifest = _manifest(tmp_path)
    text = generate_nvinfer_config(
        manifest,
        engine_path=engine,
        parser_library_path=tmp_path / "libnovasight_parser.so",
    )

    assert "network-mode=2" in text
    assert "output-tensor-meta=0" in text
    assert "parse-bbox-func-name=NvDsInferParseNovaSightRaw" in text
    assert "cluster-mode=2" in text
    assert "num-detected-classes=2" in text


def test_nvinfer_config_rejects_incomplete_model_nms_contract(tmp_path: Path) -> None:
    engine, manifest = _manifest(tmp_path)
    manifest = SimpleNamespace(
        **{
            **manifest.__dict__,
            "postprocess": SimpleNamespace(
                **{**manifest.postprocess.__dict__, "parser": "efficientnms"}
            ),
        }
    )

    with pytest.raises(ValueError, match="boxes/scores/classes output format"):
        generate_nvinfer_config(
            manifest,
            engine_path=engine,
            parser_library_path=tmp_path / "libnovasight_parser.so",
        )


def test_object_meta_is_scaled_from_model_surface_to_roi(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    object_meta = SimpleNamespace(
        class_id=1,
        confidence=0.9,
        rect_params=SimpleNamespace(left=32.0, top=64.0, width=64.0, height=32.0),
    )
    frame_meta = SimpleNamespace(obj_meta_list=SimpleNamespace(data=object_meta, next=None))
    pyds = SimpleNamespace(NvDsObjectMeta=SimpleNamespace(cast=lambda value: value))

    detections = backend._object_meta_detections(pyds, frame_meta)

    assert len(detections) == 1
    assert detections[0].cls == 1
    assert detections[0].box.x1 == pytest.approx(48.0)
    assert detections[0].box.y1 == pytest.approx(96.0)
    assert detections[0].box.x2 == pytest.approx(144.0)
    assert detections[0].box.y2 == pytest.approx(144.0)


def test_object_meta_iteration_accepts_pyds_stop_iteration_tail(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    object_meta = SimpleNamespace(
        class_id=0,
        confidence=0.8,
        rect_params=SimpleNamespace(left=1.0, top=2.0, width=3.0, height=4.0),
    )

    class TailNode:
        data = object_meta

        @property
        def next(self):
            raise StopIteration

    frame_meta = SimpleNamespace(obj_meta_list=TailNode())
    pyds = SimpleNamespace(NvDsObjectMeta=SimpleNamespace(cast=lambda value: value))

    detections = backend._object_meta_detections(pyds, frame_meta)

    assert len(detections) == 1


def test_deepstream_status_exposes_ui_metrics_without_cpu_preview_contract(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    now_ns = time.monotonic_ns()
    backend._dependency_status = DeepStreamDependencyStatus(True)
    backend._running = True
    backend._started_at_ns = now_ns - 2_000_000_000
    backend._input_frames = 3
    backend._output_buffers = 2
    backend._published_batches = 2
    backend._object_meta_frames = 2
    backend._last_frame_id = 17
    backend._last_capture_ts_ns = now_ns - 4_000_000
    backend._last_capture_interval_ms = 8.33
    backend._input_frame_samples.extend([(now_ns - 20_000_000, 0.0), (now_ns, 0.0)])
    backend._output_samples.extend([(now_ns - 20_000_000, 0.0), (now_ns, 0.0)])
    backend._publish_samples.append((now_ns, 4.0))

    status = backend.status()

    assert status["loaded"] is True
    assert status["configured"] is True
    assert status["preview_active"] is False
    assert status["inference_phase"] == "publishing"
    assert status["inference_reason"] == "DetectionBatch is being published"
    assert status["input_fps"] == 2.0
    assert status["output_fps"] == 2.0
    assert status["published_fps"] == 1.0
    assert status["latest_frame_age_ms"] == pytest.approx(4.0, abs=2.0)
    assert status["capture_profile"] == {
        "device": "/dev/video0",
        "pixel_format": "MJPG",
        "width": 1920,
        "height": 1080,
        "fps": 120,
    }
    assert status["roi_region"] == {
        "x": 720,
        "y": 300,
        "width": 480,
        "height": 480,
    }
    assert status["nvmm_output"] == {
        "width": 320,
        "height": 320,
        "pixel_format": "NV12",
        "memory": "NVMM",
    }
    assert status["postprocess"]["parser_preset"] == "auto"
    assert status["postprocess"]["compatibility"] == "yolov8_yolo11"
    assert status["postprocess"]["parser_function"] == "NvDsInferParseNovaSightRaw"
    assert status["postprocess"]["nms_owner"] == "deepstream"
    assert "preview" not in status


def test_deepstream_status_computes_percentiles_outside_stream_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    backend._dependency_status = DeepStreamDependencyStatus(True)
    now_ns = time.monotonic_ns()
    backend._publish_samples.append((now_ns, 4.0))
    lock_states: list[bool] = []
    real_sample_stats = backend_module._sample_stats

    def observed_sample_stats(samples):
        lock_states.append(backend._lock.locked())
        return real_sample_stats(samples)

    monkeypatch.setattr(backend_module, "_sample_stats", observed_sample_stats)

    backend.status()

    assert lock_states == [False, False, False, False]


def test_parser_telemetry_is_sampled_at_low_frequency_on_frame_path(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    calls = 0

    class Telemetry:
        def snapshot(self):
            nonlocal calls
            calls += 1
            return {"decode_calls": calls}

    backend._parser_telemetry = Telemetry()

    first = backend._parser_status_snapshot(now_ns=1_000_000_000)
    second = backend._parser_status_snapshot(now_ns=1_050_000_000)
    third = backend._parser_status_snapshot(now_ns=1_100_000_000)

    assert first["decode_calls"] == 1
    assert second["decode_calls"] == 1
    assert third["decode_calls"] == 2
    assert calls == 2


def test_missing_hardware_preview_encoder_does_not_disable_inference(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    Gst = SimpleNamespace(
        ElementFactory=SimpleNamespace(
            find=lambda name: None if name == "nvjpegenc" else object()
        )
    )

    backend._degrade_preview_if_unavailable(Gst)

    assert backend.pipeline_config.preview_enabled is False
    assert "nvjpegenc" not in backend.pipeline_description
    assert "nvinfer name=primary-infer" in backend.pipeline_description
    assert "missing GStreamer element(s): nvjpegenc" in backend._preview_disabled_reason


def test_deepstream_preview_valve_pauses_encoding_without_stopping_inference(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )

    class Valve:
        drop = False

        def set_property(self, name: str, value: bool) -> None:
            assert name == "drop"
            self.drop = value

    valve = Valve()
    backend._pipeline = SimpleNamespace(
        get_by_name=lambda name: valve if name == "preview-valve" else None
    )
    backend._running = True
    backend._latest_preview_jpeg = b"jpeg"

    status = backend.set_preview_active(False)

    assert valve.drop is True
    assert backend.running is True
    assert status["preview_active"] is False
    assert status["preview_available"] is False
    assert "paused" in str(status["preview_reason"])


def test_preview_stream_disconnect_does_not_override_user_preview_choice(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )

    class Valve:
        drop = True

        def set_property(self, name: str, value: bool) -> None:
            assert name == "drop"
            self.drop = value

    valve = Valve()
    backend._pipeline = SimpleNamespace(
        get_by_name=lambda name: valve if name == "preview-valve" else None
    )
    backend._running = True
    backend.set_preview_active(True)

    backend.acquire_preview_consumer()
    backend.release_preview_consumer()

    assert valve.drop is False
    assert backend.preview_active is True
    assert backend._preview_consumers == 0


def test_deepstream_backend_auto_builds_missing_parser_before_dependency_check(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _engine, manifest = _manifest(tmp_path)
    parser_path = tmp_path / "build" / "libnovasight_parser.so"
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=parser_path,
        max_publish_age_ms=55.0,
    )
    calls: list[Path] = []

    def build_parser(path: Path) -> Path:
        resolved = Path(path)
        calls.append(resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(b"parser")
        return resolved

    monkeypatch.setattr(
        "novasight.deepstream.backend.ensure_deepstream_parser_library",
        build_parser,
    )

    backend._ensure_parser_library()

    assert calls == [parser_path.resolve()]
    assert backend._parser_auto_build["attempted"] is True
    assert backend._parser_auto_build["success"] is True


def test_deepstream_publishes_batch_with_monotonic_pts_offset_fallback(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    pts_ns = 1_000_000
    now_ns = time.monotonic_ns()
    backend._running = True
    backend._inference_start_by_pts[pts_ns] = now_ns - 2_000_000
    backend._capture_ts_from_pts = lambda *_args, **_kwargs: now_ns - 4_000_000
    backend._timestamp_source = "first_probe_offset_pts"
    assert backend.wait_until_ready(0.0) is False
    frame_meta = SimpleNamespace(
        buf_pts=pts_ns,
        frame_num=1,
        obj_meta_list=None,
    )

    backend._publish_frame_meta(SimpleNamespace(), frame_meta, SimpleNamespace(pts=pts_ns))

    assert backend._published_batches == 1
    assert backend.wait_until_ready(0.0) is True
    assert backend._non_monotonic_dropped_batches == 0
    batch = backend.detection_batch_mailbox.acquire_latest(
        after_generation=-1,
        timeout_s=0.0,
    )
    assert batch is not None
    assert batch.detections == []
    assert batch.metadata["timestamp_source"] == "first_probe_offset_pts"


def test_deepstream_uses_sink_timing_when_frame_meta_pts_does_not_match(tmp_path: Path) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    input_pts_ns = 900_000_000
    stale_frame_meta_pts_ns = 100_000_000
    now_ns = time.monotonic_ns()
    backend._running = True
    backend._inference_start_by_pts[input_pts_ns] = now_ns - 2_000_000
    backend._timestamp_source = "first_probe_offset_pts"
    backend._capture_ts_from_pts = lambda pts_ns, **_kwargs: (
        now_ns - 4_000_000
        if pts_ns == input_pts_ns
        else now_ns - 723_000_000
    )
    frame_meta = SimpleNamespace(
        buf_pts=stale_frame_meta_pts_ns,
        frame_num=1,
        obj_meta_list=None,
    )

    backend._publish_frame_meta(
        SimpleNamespace(),
        frame_meta,
        SimpleNamespace(pts=input_pts_ns),
    )

    assert backend._published_batches == 1
    batch = backend.detection_batch_mailbox.acquire_latest(
        after_generation=-1,
        timeout_s=0.0,
    )
    assert batch is not None
    assert batch.result_age_ms < 55.0
    assert batch.metadata["timestamp_correlation"] == "buffer_pts"


def test_deepstream_rejects_unmatched_output_instead_of_pairing_oldest_input(
    tmp_path: Path,
) -> None:
    _engine, manifest = _manifest(tmp_path)
    backend = DeepStreamObjectBackend(
        pipeline_config=_pipeline_config(tmp_path),
        manifest=manifest,
        parser_library_path=tmp_path / "libnovasight_parser.so",
        max_publish_age_ms=55.0,
    )
    backend._inference_start_by_pts[100] = time.monotonic_ns() - 2_000_000

    timing, correlation = backend._take_input_timing(
        buffer_pts_ns=200,
        frame_meta_pts_ns=300,
        observed_ns=time.monotonic_ns(),
    )

    assert timing is None
    assert correlation == "missing"
    assert backend._timestamp_correlation_misses == 1
    assert 100 in backend._inference_start_by_pts


def test_detection_batch_mailbox_replaces_old_generation() -> None:
    mailbox = DetectionBatchMailbox()

    def batch(generation: int) -> DetectionBatch:
        ts = 1_000_000 + generation
        return DetectionBatch(
            frame_id=generation,
            generation=generation,
            capture_ts_ns=ts,
            inference_start_ts_ns=ts + 1,
            inference_end_ts_ns=ts + 2,
            publish_ts_ns=ts + 3,
            roi_size=(480, 480),
            model_input_size=(320, 320),
        )

    assert mailbox.publish(batch(1)) is True
    assert mailbox.publish(batch(3)) is True
    assert mailbox.publish(batch(2)) is False
    latest = mailbox.acquire_latest(after_generation=1, timeout_s=0.0)

    assert latest is not None
    assert latest.generation == 3
    assert mailbox.status()["pending_depth"] == 1
    assert mailbox.status()["overwritten_batches"] == 1


def test_deepstream_runtime_hands_only_detection_batch_to_shared_control() -> None:
    mailbox = DetectionBatchMailbox()
    batch = DetectionBatch(
        frame_id=7,
        generation=3,
        capture_ts_ns=1_000_000,
        inference_start_ts_ns=1_000_100,
        inference_end_ts_ns=1_000_200,
        publish_ts_ns=1_000_300,
        roi_size=(480, 480),
        model_input_size=(320, 320),
    )
    mailbox.publish(batch)
    received = []
    backend = SimpleNamespace(
        detection_batch_mailbox=mailbox,
        running=True,
        pipeline_config=SimpleNamespace(
            roi_width=480,
            roi_height=480,
            capture_width=1920,
            capture_height=1080,
            roi_left=720,
            roi_top=300,
        ),
    )
    runtime = SimpleNamespace(running=False)
    pipeline = DeepStreamRuntimePipeline(backend=backend, runtime=runtime)

    def process_detection_batch(value, **geometry):
        received.append((value, geometry))
        pipeline._stop.set()
        return SimpleNamespace(observation_updated=True)

    runtime.process_detection_batch = process_detection_batch
    pipeline._detection_loop()

    assert received[0][0] is batch
    assert received[0][1] == {
        "width": 480,
        "height": 480,
        "source_width": 1920,
        "source_height": 1080,
        "roi_offset_x": 720,
        "roi_offset_y": 300,
    }
    assert pipeline.stats.control_observations == 1


def test_deepstream_stop_disconnects_backend_before_clearing_runtime_session() -> None:
    events: list[str] = []
    backend = SimpleNamespace(
        stop=lambda: events.append("backend.stop"),
    )
    runtime = SimpleNamespace(
        running=True,
        reset_runtime_session=lambda reason: events.append(f"runtime.reset:{reason}"),
    )
    pipeline = DeepStreamRuntimePipeline(backend=backend, runtime=runtime)
    pipeline.stats.processed_frames = 12
    pipeline.stats.control_observations = 9

    pipeline.stop()

    assert events == ["backend.stop", "runtime.reset:RUNTIME_STOPPED"]
    assert pipeline.stats.processed_frames == 0
    assert pipeline.stats.control_observations == 0


@pytest.mark.parametrize(
    ("source", "inference_enabled", "message"),
    [
        ("image", True, "source.default=capture"),
        ("capture", False, "consumers.inference=true"),
    ],
)
def test_deepstream_runtime_rejects_unsupported_ui_modes(
    source: str,
    inference_enabled: bool,
    message: str,
) -> None:
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            source=SimpleNamespace(default=source),
            consumers=SimpleNamespace(inference=inference_enabled),
        ),
        models=None,
    )

    with pytest.raises(RuntimeError, match=message):
        create_deepstream_runtime_pipeline(runtime=runtime)


def test_deepstream_runtime_generates_missing_manifest_from_engine_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "models")
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project.id,
        "v1",
        "onnx",
        "demo.engine",
        ["body", "head"],
        "1x3x640x640",
    )
    engine_path = registry.data_dir / project.name / version.version / "demo.engine"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(b"engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        engine_path.name,
        "legacy-checksum",
        "ready",
    )
    registry.publish(project.id, artifact.id)
    config = RuntimeConfig()
    config.source.default = "capture"
    config.inference.backend = "deepstream_nvinfer"
    config.capture.backend = "deepstream_nvinfer"
    config.capture.memory = "nvmm"
    config.capture.pixel_format = "MJPG"
    config.capture.width = 1920
    config.capture.height = 1080
    config.capture.fps = 120
    config.roi.size = 480
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "input_name": "images",
            "input_shape": "1x3x256x256",
            "input_dtype": "float32",
            "output_name": "output0",
            "output_shape": "1x6x1344",
            "output_dtype": "float32",
        }
    )
    runtime = SimpleNamespace(config=config, models=registry, inference=inference)

    pipeline = create_deepstream_runtime_pipeline(runtime=runtime)

    manifest_path = engine_path.with_name(f"{engine_path.name}.manifest.json")
    manifest = read_manifest(manifest_path)
    runtime.prepared_engine_manifest = PreparedEngineManifest.create(
        engine_path=engine_path,
        manifest=manifest,
    )
    monkeypatch.setattr(
        runtime_pipeline_module,
        "ensure_engine_manifest",
        lambda *_args, **_kwargs: pytest.fail(
            "model-switch handoff must not revalidate an unchanged Engine"
        ),
    )
    restarted_pipeline = create_deepstream_runtime_pipeline(runtime=runtime)
    assert pipeline.backend.pipeline_config.nvinfer_config_path.name == "active-nvinfer.ini"
    assert pipeline.backend.manifest.model_fingerprint == manifest.model_fingerprint
    assert restarted_pipeline.backend.manifest.model_fingerprint == manifest.model_fingerprint
    assert not engine_path.with_name("model.manifest.json").exists()
    assert manifest.input.shape == [1, 3, 256, 256]
    assert manifest.output.shape == [1, 6, 1344]
    assert manifest.output.class_names == ["body", "head"]
    assert registry.get_artifact(artifact.id).status == "ready"
    assert registry.get_artifact(artifact.id).checksum == manifest.artifact.sha256
    assert registry.get_version(version.id).input_shape == "1x3x256x256"


def test_existing_manifest_is_rebuilt_when_engine_tensor_contract_changed(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "demo.engine"
    engine_path.write_bytes(b"engine")
    stale_manifest = build_engine_manifest(
        model_id="demo",
        display_name="demo",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 640, 640], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 6, 1344], "float32", "NCHW"),
        class_count=2,
        class_names=["body", "head"],
        output_has_objectness=False,
        validated=True,
    )
    write_manifest(stale_manifest, engine_path.with_name("model.manifest.json"))
    write_manifest(
        stale_manifest,
        engine_path.with_name(f"{engine_path.name}.manifest.json"),
    )
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "input_name": "input_tensor",
            "input_shape": "1x3x320x512",
            "input_dtype": "float16",
            "output_name": "output0",
            "output_shape": "1x6x1344",
            "output_dtype": "float32",
            "outputs": {
                "output0": {
                    "shape": [1, 6, 1344],
                    "dtype": "float32",
                }
            },
        }
    )

    manifest, regenerated = ensure_engine_manifest(
        inference,
        engine_path=engine_path,
        model_id="demo",
        display_name="demo",
        classes=["body", "head"],
        registered_input_shape="1x3x640x640",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )

    assert regenerated is True
    manifest = read_manifest(engine_path.with_name(f"{engine_path.name}.manifest.json"))
    assert manifest.input.name == "input_tensor"
    assert manifest.input.shape == [1, 3, 320, 512]
    assert manifest.input.dtype == "float16"
    assert engine_path.with_name(f"{engine_path.name}.manifest.json").is_file()
    assert not engine_path.with_name("model.manifest.json").exists()


def test_invalid_legacy_manifest_is_ignored_and_preserved(tmp_path: Path) -> None:
    engine_path = tmp_path / "demo.engine"
    engine_path.write_bytes(b"engine")
    legacy_manifest_path = engine_path.with_name("model.manifest.json")
    legacy_manifest_path.write_text("{}", encoding="utf-8")

    assert remove_matching_legacy_manifest(engine_path) is False
    assert legacy_manifest_path.is_file()


def test_deepstream_runtime_replaces_placeholder_classes_from_raw_yolo_output(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "models")
    project = registry.create_project("demo", "")
    version = registry.create_version(
        project.id,
        "v1",
        "onnx",
        "demo.engine",
        ["target"],
        "1x3x256x256",
    )
    engine_path = registry.data_dir / project.name / version.version / "demo.engine"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(b"engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        engine_path.name,
        "legacy-checksum",
        "ready",
    )
    registry.publish(project.id, artifact.id)
    config = RuntimeConfig()
    config.source.default = "capture"
    config.inference.backend = "deepstream_nvinfer"
    config.capture.backend = "deepstream_nvinfer"
    config.capture.memory = "nvmm"
    config.capture.pixel_format = "MJPG"
    config.capture.width = 1920
    config.capture.height = 1080
    config.capture.fps = 120
    config.roi.size = 480
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "input_name": "images",
            "input_shape": "1x3x256x256",
            "input_dtype": "float32",
            "output_name": "output0",
            "output_shape": "1x8x1344",
            "output_dtype": "float32",
        }
    )
    runtime = SimpleNamespace(config=config, models=registry, inference=inference)

    pipeline = create_deepstream_runtime_pipeline(runtime=runtime)

    manifest = pipeline.backend.manifest
    assert manifest.output.class_count == 4
    assert manifest.output.class_names == ["class_0", "class_1", "class_2", "class_3"]
    assert manifest.output.has_objectness is False
    assert engine_path.with_name(f"{engine_path.name}.manifest.json").is_file()
    assert not engine_path.with_name("model.manifest.json").exists()
    assert registry.get_version(version.id).classes == manifest.output.class_names


def test_existing_v8_manifest_ignores_ambiguous_class_count_in_filename(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "0305大碗模型三类_v8s256.engine"
    engine_path.write_bytes(b"engine")
    stale_manifest = build_engine_manifest(
        model_id="demo",
        display_name="demo",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        class_names=["class_0", "class_1", "class_2", "class_3"],
        output_has_objectness=False,
        validated=True,
    )
    write_manifest(stale_manifest, engine_path.with_name("model.manifest.json"))
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "input_name": "images",
            "input_shape": "1x3x256x256",
            "input_dtype": "float32",
            "output_name": "output0",
            "output_shape": "1x8x1344",
            "output_dtype": "float32",
        }
    )

    manifest, regenerated = ensure_engine_manifest(
        inference,
        engine_path=engine_path,
        model_id="demo",
        display_name="demo",
        classes=list(stale_manifest.output.class_names),
        registered_input_shape="1x3x256x256",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )

    assert regenerated is False
    assert manifest.output.class_count == 4
    assert manifest.output.class_names == ["class_0", "class_1", "class_2", "class_3"]
    assert manifest.output.has_objectness is False
    assert engine_path.with_name(f"{engine_path.name}.manifest.json").is_file()
    assert not engine_path.with_name("model.manifest.json").exists()
    config_text = generate_nvinfer_config(
        manifest,
        engine_path=engine_path,
        parser_library_path=tmp_path / "libnovasight_parser.so",
    )
    assert "num-detected-classes=4" in config_text


def test_single_target_objectness_contract_is_not_reclassified() -> None:
    classes, has_objectness = resolve_yolo_class_contract(
        [1, 6, 1344],
        ["target"],
    )

    assert classes == ["target"]
    assert has_objectness is True


def test_explicit_v5_preset_derives_single_class_and_persists_contract(tmp_path: Path) -> None:
    engine_path = tmp_path / "fast-target.engine"
    engine_path.write_bytes(b"engine")
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "input_name": "images",
            "input_shape": "1x3x320x320",
            "input_dtype": "float16",
            "output_name": "output0",
            "output_shape": "1x6x2100",
            "output_dtype": "float32",
        }
    )

    manifest, generated = ensure_engine_manifest(
        inference,
        engine_path=engine_path,
        model_id="fast-target",
        display_name="fast-target",
        classes=["target"],
        registered_input_shape="1x3x320x320",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
        parser_preset="yolov5",
    )

    assert generated is True
    assert manifest.output.class_count == 1
    assert manifest.output.has_objectness is True
    assert manifest.postprocess.parser_preset == "yolov5"
    persisted = read_manifest(engine_path.with_name(f"{engine_path.name}.manifest.json"))
    assert persisted.postprocess.parser_preset == "yolov5"


def test_missing_manifest_is_not_generated_for_builtin_nms_output(tmp_path: Path) -> None:
    engine_path = tmp_path / "demo.engine"
    engine_path.write_bytes(b"engine")
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "input_name": "images",
            "input_shape": "1x3x256x256",
            "input_dtype": "float32",
            "output_name": "detections",
            "output_shape": "1x300x6",
            "output_dtype": "float32",
        }
    )

    with pytest.raises(ValueError, match="built-in Decode/NMS"):
        ensure_engine_manifest(
            inference,
            engine_path=engine_path,
            model_id="demo",
            display_name="demo",
            classes=["body", "head"],
            registered_input_shape="1x3x256x256",
            confidence_threshold=0.25,
            nms_iou_threshold=0.45,
        )

    assert not engine_path.with_name("model.manifest.json").exists()
    assert not engine_path.with_name(f"{engine_path.name}.manifest.json").exists()


def test_missing_manifest_is_not_generated_for_unrecognized_multi_output_engine(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "demo.engine"
    engine_path.write_bytes(b"engine")
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "input_name": "images",
            "input_shape": "1x3x256x256",
            "input_dtype": "float32",
            "output_name": "boxes",
            "output_shape": "1x100x4",
            "output_dtype": "float32",
            "outputs": {
                "boxes": {"shape": [1, 100, 4], "dtype": "float32"},
                "masks": {"shape": [1, 32, 64, 64], "dtype": "float32"},
            },
        }
    )

    with pytest.raises(ValueError, match="could not identify a supported detection"):
        ensure_engine_manifest(
            inference,
            engine_path=engine_path,
            model_id="demo",
            display_name="demo",
            classes=["body", "head"],
            registered_input_shape="1x3x256x256",
            confidence_threshold=0.25,
            nms_iou_threshold=0.45,
        )

    assert not engine_path.with_name("model.manifest.json").exists()
    assert not engine_path.with_name(f"{engine_path.name}.manifest.json").exists()


def test_missing_manifest_selects_unique_raw_yolo_tensor_from_four_outputs(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "demo.engine"
    engine_path.write_bytes(b"engine")
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "io_tensors": [
                {
                    "name": "images",
                    "mode": "input",
                    "shape": [1, 3, 256, 256],
                    "dtype": "float16",
                },
                {
                    "name": "predictions",
                    "mode": "output",
                    "shape": [1, 6, 1344],
                    "dtype": "float16",
                },
                {
                    "name": "prototype",
                    "mode": "output",
                    "shape": [1, 32, 64, 64],
                    "dtype": "float16",
                },
                {
                    "name": "feature_a",
                    "mode": "output",
                    "shape": [1, 16, 32, 32],
                    "dtype": "float16",
                },
                {
                    "name": "feature_b",
                    "mode": "output",
                    "shape": [1, 8, 16, 16],
                    "dtype": "float16",
                },
            ],
        }
    )

    manifest, regenerated = ensure_engine_manifest(
        inference,
        engine_path=engine_path,
        model_id="demo",
        display_name="demo",
        classes=["body", "head"],
        registered_input_shape="1x3x256x256",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )

    assert regenerated is True
    assert manifest.output.name == "predictions"
    assert manifest.output.shape == [1, 6, 1344]


def test_missing_manifest_supports_four_output_efficient_nms_engine(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "demo.engine"
    engine_path.write_bytes(b"engine")
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "io_tensors": [
                {
                    "name": "images",
                    "mode": "input",
                    "shape": [1, 3, 640, 640],
                    "dtype": "float16",
                },
                {
                    "name": "num_dets",
                    "mode": "output",
                    "shape": [1, 1],
                    "dtype": "int32",
                },
                {
                    "name": "det_boxes",
                    "mode": "output",
                    "shape": [1, 300, 4],
                    "dtype": "float32",
                },
                {
                    "name": "det_scores",
                    "mode": "output",
                    "shape": [1, 300],
                    "dtype": "float32",
                },
                {
                    "name": "det_classes",
                    "mode": "output",
                    "shape": [1, 300],
                    "dtype": "int32",
                },
            ],
        }
    )

    manifest, regenerated = ensure_engine_manifest(
        inference,
        engine_path=engine_path,
        model_id="demo",
        display_name="demo",
        classes=["body", "head"],
        registered_input_shape="1x3x640x640",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )
    nvinfer = generate_nvinfer_config(
        manifest,
        engine_path=engine_path,
        parser_library_path=tmp_path / "libnovasight_parser.so",
    )

    assert regenerated is True
    manifest = read_manifest(engine_path.with_name(f"{engine_path.name}.manifest.json"))
    assert manifest.output.format == "efficientnms_boxes_scores_classes"
    assert manifest.postprocess.parser == "efficientnms"
    assert [item.name for item in manifest.output.bindings] == [
        "num_dets",
        "det_boxes",
        "det_scores",
        "det_classes",
    ]
    assert "output-blob-names=num_dets;det_boxes;det_scores;det_classes" in nvinfer
    assert "cluster-mode=4" in nvinfer


@pytest.mark.parametrize(
    ("candidate_count", "expected_parser", "expected_function"),
    [
        (6300, "yolo", "NvDsInferParseNovaSightRaw"),
        (300, "decoded_nms", "NvDsInferParseNovaSightDecodedNms"),
    ],
)
def test_six_column_output_uses_density_to_resolve_raw_vs_decoded_contract(
    tmp_path: Path,
    candidate_count: int,
    expected_parser: str,
    expected_function: str,
) -> None:
    engine_path = tmp_path / f"six-column-{candidate_count}.engine"
    engine_path.write_bytes(b"engine")
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": True,
            "io_tensors": [
                {
                    "name": "images",
                    "mode": "input",
                    "shape": [1, 3, 640, 640],
                    "dtype": "float16",
                },
                {
                    "name": "output0",
                    "mode": "output",
                    "shape": [1, candidate_count, 6],
                    "dtype": "float16",
                },
            ],
        }
    )

    manifest, _ = ensure_engine_manifest(
        inference,
        engine_path=engine_path,
        model_id="six-column",
        display_name="Six column",
        classes=["enemy"],
        registered_input_shape="1x3x640x640",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )
    nvinfer = generate_nvinfer_config(
        manifest,
        engine_path=engine_path,
        parser_library_path=tmp_path / "libnovasight_parser.so",
    )

    assert manifest.postprocess.parser == expected_parser
    assert f"parse-bbox-func-name={expected_function}" in nvinfer


def test_missing_manifest_supports_rockchip_yolov5_three_scale_heads(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "rockchip-yolov5.engine"
    engine_path.write_bytes(b"engine")
    outputs = {
        "output0": {"shape": [1, 21, 40, 40], "dtype": "float16"},
        "286": {"shape": [1, 21, 20, 20], "dtype": "float16"},
        "288": {"shape": [1, 21, 10, 10], "dtype": "float16"},
    }
    inference = SimpleNamespace(
        probe=lambda *_args: {
            "loaded": False,
            "reason": (
                "unsupported TensorRT detection output contract; decoder expects one "
                f"NxC/CxN tensor with at least 5 columns, outputs={outputs}"
            ),
            "io_tensors": [
                {
                    "name": "images",
                    "mode": "input",
                    "shape": [1, 3, 320, 320],
                    "dtype": "float16",
                },
                *[
                    {
                        "name": name,
                        "mode": "output",
                        **contract,
                    }
                    for name, contract in outputs.items()
                ],
            ],
        }
    )

    manifest, regenerated = ensure_engine_manifest(
        inference,
        engine_path=engine_path,
        model_id="rockchip-yolov5",
        display_name="Rockchip YOLOv5",
        classes=["body", "head"],
        registered_input_shape="1x3x320x320",
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
    )
    nvinfer = generate_nvinfer_config(
        manifest,
        engine_path=engine_path,
        parser_library_path=tmp_path / "libnovasight_parser.so",
    )

    assert regenerated is True
    assert manifest.output.format == "rockchip_yolov5_three_scale"
    assert manifest.postprocess.parser == "rockchip_yolov5"
    assert [item.name for item in manifest.output.bindings] == ["output0", "286", "288"]
    assert manifest.output.strides == [8, 16, 32]
    assert manifest.output.anchors == [
        [10.0, 13.0, 16.0, 30.0, 33.0, 23.0],
        [30.0, 61.0, 62.0, 45.0, 59.0, 119.0],
        [116.0, 90.0, 156.0, 198.0, 373.0, 326.0],
    ]
    assert "output-blob-names=output0;286;288" in nvinfer
    assert "cluster-mode=2" in nvinfer
