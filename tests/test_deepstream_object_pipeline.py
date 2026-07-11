import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from novasight.contracts import DetectionBatch
from novasight.deepstream.backend import DeepStreamDependencyStatus, DeepStreamObjectBackend
from novasight.deepstream.nvinfer_config import generate_nvinfer_config
from novasight.deepstream.pipeline_builder import (
    DeepStreamPipelineConfig,
    build_deepstream_pipeline,
)
from novasight.model_registry.manifest import TensorSpec, build_engine_manifest
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


def test_deepstream_pipeline_is_nvmm_latest_only_without_cpu_image_sink(tmp_path: Path) -> None:
    pipeline = build_deepstream_pipeline(_pipeline_config(tmp_path))

    assert pipeline.count("max-size-buffers=1") == 3
    assert pipeline.count("leaky=downstream") == 3
    assert "nvv4l2decoder mjpeg=1" in pipeline
    assert "nvvidconv left=720 right=1200 top=300 bottom=780" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=320,height=320" in pipeline
    assert "nvstreammux name=mux batch-size=1 live-source=1" in pipeline
    assert "nvinfer name=primary-infer" in pipeline
    assert "appsink" not in pipeline
    assert "videoconvert" not in pipeline
    assert "video/x-raw,format=BGR" not in pipeline


def test_nvinfer_config_uses_native_decode_and_exactly_one_nms(tmp_path: Path) -> None:
    engine, manifest = _manifest(tmp_path)
    text = generate_nvinfer_config(
        manifest,
        engine_path=engine,
        parser_library_path=tmp_path / "libnovasight_parser.so",
    )

    assert "network-mode=2" in text
    assert "output-tensor-meta=1" in text
    assert "parse-bbox-func-name=NvDsInferParseNovaSight" in text
    assert "cluster-mode=2" in text
    assert "num-detected-classes=2" in text


def test_nvinfer_config_rejects_model_nms_without_explicit_contract(tmp_path: Path) -> None:
    engine, manifest = _manifest(tmp_path)
    manifest = SimpleNamespace(
        **{
            **manifest.__dict__,
            "postprocess": SimpleNamespace(
                **{**manifest.postprocess.__dict__, "parser": "efficientnms"}
            ),
        }
    )

    with pytest.raises(ValueError, match="built-in Decode/NMS"):
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
    backend._published_batches = 2
    backend._object_meta_frames = 2
    backend._last_frame_id = 17
    backend._last_capture_ts_ns = now_ns - 4_000_000
    backend._last_capture_interval_ms = 8.33
    backend._input_frame_samples.extend([(now_ns - 20_000_000, 0.0), (now_ns, 0.0)])
    backend._publish_samples.append((now_ns, 4.0))

    status = backend.status()

    assert status["input_fps"] == 2.0
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
    assert "preview" not in status


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
