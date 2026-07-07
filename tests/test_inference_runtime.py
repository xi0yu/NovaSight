"""Tests for the inference runtime and its contract with the runtime service.

The runtime is a defensive layer: it must convert broken engine responses,
exceptions, and missing inference into an empty FrameContext so that
downstream plugin control stays safe. The merged test covers that contract
once instead of repeating it across six near-identical tests.
"""
import copy
import builtins
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from novasight.capture.source import CapturedFrame, FrameResource
from novasight.config import RuntimeConfig
from novasight.contracts import Detection, FrameContext, Track
from novasight.coordinates import CoordinateTransform
from novasight.executors import ExecutorRegistry
from novasight.control import CommandScheduler, ControlOutput
from novasight.executors.contracts import ExecutionResult
from novasight.inference import (
    InferenceDetection,
    InferenceResult,
    InferenceRuntime,
    OnnxRuntimeInferenceEngine,
    TensorRtInferenceEngine,
    UnavailableInferenceEngine,
)
from novasight.inference.input import (
    PreparedTensorInput,
    TensorInputShape,
    parse_tensor_input_shape,
    prepare_tensor_input,
)
from novasight.inference.onnxruntime_engine import (
    _preprocess_debug,
    map_model_detections_to_roi_frame,
)
from novasight.inference.postprocess.yolo import decode_nx6_detections
from novasight.deepstream.backend import DeepStreamDetectionBackend
from novasight.deepstream.pipeline_builder import DeepStreamPipelineConfig, build_deepstream_pipeline
from novasight.deepstream.tensor_meta import output_tensor_to_detection_batch
from novasight.model_registry.deepstream_config import (
    DEEPSTREAM_CONFIG_TEMPLATE_VERSION,
    generate_nvinfer_config,
)
from novasight.model_registry.fingerprint import sha256_file
from novasight.model_registry.manifest import (
    ModelManifest,
    TensorSpec,
    build_engine_manifest,
    read_manifest,
    write_manifest,
)
from novasight.model_registry.scanner import scan_model_artifacts
from novasight.inference.preprocess import (
    DeviceTensor,
    GPU_RESOURCE_PREPROCESS_NOT_IMPLEMENTED,
    TensorPreprocessError,
    TensorPreprocessResult,
    prepare_host_tensor,
    prepare_tensor,
)
from novasight.inference.jetson import (
    DEFAULT_JETSON_GPU_RESOURCE_BRIDGE_MODULE,
    JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
    JETSON_GPU_RESOURCE_BRIDGE_ENV,
    JETSON_GPU_RESOURCE_BRIDGE_INVALID,
    JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_PAYLOAD_FIELDS,
    JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_RESULT_FIELDS,
    JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE,
    JetsonGpuResourcePreprocessor,
)


def _ctypes_native_payload(**overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        "resource_handle": object(),
        "frame_id": 17,
        "capture_ts_ns": 1_234_567_890,
        "resource_kind": "gstreamer_sample",
        "resource_memory": "dmabuf",
        "resource_source": "appsink",
        "dmabuf_fd": 42,
        "resource_metadata": {},
        "resource_width": 320,
        "resource_height": 320,
        "resource_pixel_format": "NV12",
        "width": 320,
        "height": 320,
        "pixel_format": "NV12",
        "source_width": 1920,
        "source_height": 1080,
        "roi_offset_x": 800,
        "roi_offset_y": 380,
        "needs_resize": False,
        "model_shape": {"batch": 1, "channels": 3, "height": 320, "width": 320},
        "nchw": (1, 3, 320, 320),
        "dtype": "float32",
    }
    payload.update(overrides)
    return payload
from novasight.model_registry import ModelRegistry

from novasight.runtime.aim import (
    AimPointConfig,
    AimPointGenerator,
    EstimatedTargetState,
    LatencyCompensationConfig,
    LatencyCompensator,
)
from novasight.runtime.candidates import (
    CandidateFilter,
    CandidateFilterConfig,
    QualityScoreConfig,
    QualityScorer,
    RatioCheckConfig,
    SelectionFovConfig,
)
from novasight.runtime.kalman import KalmanConfig, KalmanEstimator
from novasight.runtime.target_selector import RuntimeTargetSelector
from novasight.runtime.tracker import RuntimeTracker, TrackerConfig
from novasight.runtime import RuntimeService
from novasight.runtime.recorder import CONTROL_FRAME_FIELDS, ControlFrameCsvRecorder
from novasight.runtime.replay import (
    ControlFrameReplay,
    ReplayAcceptanceCase,
    ReplayAcceptanceGate,
    ReplayInjection,
    build_default_replay_acceptance_cases,
    compare_replay_metrics,
    run_replay_acceptance,
)


def test_unavailable_engine_reports_reason_and_returns_empty_result() -> None:
    engine = UnavailableInferenceEngine("TensorRT unavailable")

    assert engine.available() is False
    assert engine.status() == {
        "selected": "unavailable",
        "available": False,
        "loaded": False,
        "reason": "TensorRT unavailable",
    }

    result = engine.infer(CapturedFrame(1, 640, 480, "BGR", 123, 1.0, image=None))

    assert result.available is False
    assert result.detections == []
    assert result.reason == "TensorRT unavailable"


def test_tensorrt_engine_rejects_non_engine_artifact() -> None:
    engine = TensorRtInferenceEngine()

    with pytest.raises(ValueError, match="must be .engine"):
        engine.load(Path("model.onnx"), classes=["target"], input_shape="1x3x640x640")


def test_tensorrt_engine_reports_not_loaded_before_infer() -> None:
    engine = TensorRtInferenceEngine()

    result = engine.infer(CapturedFrame(1, 640, 480, "BGR", 123, 1.0, image=None))

    assert result.available is False
    assert result.reason == "TensorRT engine not loaded"


def test_tensor_input_shape_parser_accepts_nchw_shapes() -> None:
    shape = parse_tensor_input_shape("1x3x640x640")

    assert shape.batch == 1
    assert shape.channels == 3
    assert shape.height == 640
    assert shape.width == 640


def test_shared_yolo_parser_decodes_channel_first_output_with_class_aware_nms() -> None:
    import numpy as np

    # output0 [1, 8, N] = [cx, cy, w, h, class0..class3] x N.
    output = np.zeros((1, 8, 10), dtype=np.float32)
    output[0, :, 0] = [100, 120, 40, 60, 0.92, 0.1, 0.0, 0.0]
    output[0, :, 1] = [102, 122, 40, 60, 0.88, 0.05, 0.0, 0.0]
    output[0, :, 2] = [200, 210, 30, 50, 0.1, 0.94, 0.0, 0.0]
    output[0, :, 3] = [300, 310, 30, 40, 0.2, 0.1, 0.24, 0.0]
    debug: dict[str, object] = {}

    detections = decode_nx6_detections(
        output,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        class_count=4,
        debug=debug,
    )

    assert [item.cls for item in detections] == [1, 0]
    assert [round(item.score, 2) for item in detections] == [0.94, 0.92]
    assert detections[0].x1 == pytest.approx(185.0)
    assert detections[0].y1 == pytest.approx(185.0)
    assert detections[0].x2 == pytest.approx(215.0)
    assert detections[0].y2 == pytest.approx(235.0)
    assert detections[1].x1 == pytest.approx(80.0)
    assert detections[1].y1 == pytest.approx(90.0)
    assert detections[1].x2 == pytest.approx(120.0)
    assert detections[1].y2 == pytest.approx(150.0)
    assert debug["selected_layout"] == "yolov8-cxcywh-cls"
    assert debug["threshold_candidates"] == 3
    assert debug["nms_detections"] == 2


def test_model_manifest_records_engine_identity_and_roundtrips(tmp_path) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake engine bytes")

    manifest = build_engine_manifest(
        model_id="player_detector_v1",
        display_name="Player Detector V1",
        engine_path=engine_path,
        input_spec=TensorSpec(
            name="images",
            shape=[1, 3, 256, 256],
            dtype="float32",
            layout="NCHW",
        ),
        output_spec=TensorSpec(
            name="output0",
            shape=[1, 8, 1344],
            dtype="float32",
            layout="NCHW",
        ),
        class_count=4,
    )
    manifest_path = tmp_path / "model.manifest.json"

    write_manifest(manifest, manifest_path)
    loaded = read_manifest(manifest_path)

    assert isinstance(loaded, ModelManifest)
    assert loaded.schema_version == 1
    assert loaded.model_id == "player_detector_v1"
    assert loaded.artifact.engine_path == "model.engine"
    assert loaded.artifact.sha256 == sha256_file(engine_path)
    assert loaded.artifact.size_bytes == len(b"fake engine bytes")
    assert loaded.input.name == "images"
    assert loaded.output.name == "output0"
    assert loaded.output.class_count == 4
    assert loaded.postprocess.parser == "yolo"
    assert loaded.deepstream.output_tensor_meta is True
    assert loaded.model_fingerprint == manifest.model_fingerprint


def test_deepstream_config_is_generated_from_manifest(tmp_path) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake engine bytes")
    manifest = build_engine_manifest(
        model_id="player_detector_v1",
        display_name="Player Detector V1",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
    )

    text, fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)

    assert "[property]" in text
    assert f"model-engine-file={engine_path.resolve()}" in text
    assert "batch-size=1" in text
    assert "network-mode=2" in text
    assert "network-type=100" in text
    assert "output-tensor-meta=1" in text
    assert "output-blob-names=output0" in text
    assert "net-scale-factor=0.00392156862745098" in text
    assert fingerprint
    assert DEEPSTREAM_CONFIG_TEMPLATE_VERSION in fingerprint


def test_model_scanner_reports_need_confirm_ready_and_invalid(tmp_path) -> None:
    unconfigured = tmp_path / "unconfigured.engine"
    unconfigured.write_bytes(b"unconfigured")
    configured_dir = tmp_path / "configured"
    configured_dir.mkdir()
    configured = configured_dir / "model.engine"
    configured.write_bytes(b"configured")
    manifest = build_engine_manifest(
        model_id="configured",
        display_name="Configured",
        engine_path=configured,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, configured_dir / "model.manifest.json")
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    invalid = invalid_dir / "model.engine"
    invalid.write_bytes(b"first")
    invalid_manifest = build_engine_manifest(
        model_id="invalid",
        display_name="Invalid",
        engine_path=invalid,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
    )
    write_manifest(invalid_manifest, invalid_dir / "model.manifest.json")
    invalid.write_bytes(b"changed after manifest")

    results = scan_model_artifacts(tmp_path)
    unconfigured_status = next(item for item in results if item.path == unconfigured)
    configured_status = next(item for item in results if item.path == configured)
    invalid_status = next(item for item in results if item.path == invalid)

    assert unconfigured_status.status == "need_confirm"
    assert unconfigured_status.reason == "model manifest is missing"
    assert invalid_status.status == "invalid"
    assert invalid_status.reason == "model manifest artifact sha256 does not match file"
    assert configured_status.status == "ready"
    assert configured_status.manifest_path == configured_dir / "model.manifest.json"
    assert configured_status.deepstream_config_path == configured_dir / "deepstream.ini"
    assert configured_status.model_fingerprint == manifest.model_fingerprint


def test_deepstream_pipeline_builder_uses_nvmm_nvinfer_and_leaky_queues(tmp_path) -> None:
    config_path = tmp_path / "deepstream.ini"
    config_path.write_text("[property]\n", encoding="utf-8")

    pipeline = build_deepstream_pipeline(
        DeepStreamPipelineConfig(
            device="/dev/video0",
            capture_width=1920,
            capture_height=1080,
            fps=120,
            roi_left=720,
            roi_top=300,
            roi_size=480,
            model_width=256,
            model_height=256,
            nvinfer_config_path=config_path,
        )
    )

    assert "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true" in pipeline
    assert "image/jpeg,width=1920,height=1080,framerate=120/1" in pipeline
    assert pipeline.count("queue max-size-buffers=1 leaky=downstream") >= 3
    assert "nvv4l2decoder mjpeg=1" in pipeline
    assert "video/x-raw(memory:NVMM),format=I420" in pipeline
    assert "nvvidconv left=720 right=1200 top=300 bottom=780" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=256,height=256" in pipeline
    assert "nvstreammux name=mux batch-size=1 width=256 height=256 live-source=1" in pipeline
    assert f"nvinfer name=primary-infer config-file-path={config_path.resolve()}" in pipeline
    assert "video/x-raw,format=BGR" not in pipeline
    assert "videoconvert" not in pipeline


def test_deepstream_output_tensor_reuses_shared_parser_and_returns_detection_batch(tmp_path) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    manifest = build_engine_manifest(
        model_id="combat",
        display_name="Combat",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
    )
    import numpy as np

    output = np.zeros((1, 8, 1344), dtype=np.float32)
    output[0, :, 0] = [128, 128, 32, 64, 0.9, 0.1, 0.0, 0.0]

    batch = output_tensor_to_detection_batch(
        output,
        manifest=manifest,
        frame_id=12,
        capture_ts_ns=1000,
        inference_start_ts_ns=1200,
        inference_end_ts_ns=1800,
        roi_width=480,
        roi_height=480,
    )

    assert batch.frame_id == 12
    assert batch.capture_ts_ns == 1000
    assert batch.coordinate_space == "roi"
    assert batch.classes == ["0", "1", "2", "3"]
    assert len(batch.detections) == 1
    detection = batch.detections[0]
    assert detection.cls == 0
    assert detection.score == pytest.approx(0.9)
    assert detection.x1 == pytest.approx(210.0)
    assert detection.y1 == pytest.approx(180.0)
    assert detection.x2 == pytest.approx(270.0)
    assert detection.y2 == pytest.approx(300.0)


def test_deepstream_output_tensor_rejects_manifest_shape_mismatch(tmp_path) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    manifest = build_engine_manifest(
        model_id="combat",
        display_name="Combat",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
    )
    import numpy as np

    output = np.zeros((1, 7, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="tensor shape"):
        output_tensor_to_detection_batch(
            output,
            manifest=manifest,
            frame_id=12,
            capture_ts_ns=1000,
            inference_start_ts_ns=1200,
            inference_end_ts_ns=1800,
            roi_width=480,
            roi_height=480,
        )


def test_deepstream_backend_reports_missing_runtime_dependencies(tmp_path, monkeypatch) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    config_path = tmp_path / "deepstream.ini"
    config_path.write_text("[property]\n", encoding="utf-8")
    manifest = build_engine_manifest(
        model_id="combat",
        display_name="Combat",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
    )
    backend = DeepStreamDetectionBackend(
        pipeline_config=DeepStreamPipelineConfig(
            device="/dev/video0",
            capture_width=1920,
            capture_height=1080,
            fps=120,
            roi_left=720,
            roi_top=300,
            roi_size=480,
            model_width=256,
            model_height=256,
            nvinfer_config_path=config_path,
        ),
        manifest=manifest,
        roi_width=480,
        roi_height=480,
    )

    def import_module(name: str):
        if name == "gi":
            raise ModuleNotFoundError("No module named 'gi'")
        return __import__(name)

    monkeypatch.setattr("novasight.deepstream.backend.importlib.import_module", import_module)

    status = backend.dependency_status()
    assert status.available is False
    assert status.reason == "gstreamer-python-unavailable"
    with pytest.raises(RuntimeError, match="DeepStream backend unavailable"):
        backend.start()


def test_deepstream_backend_publishes_latest_detection_batch(tmp_path) -> None:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    config_path = tmp_path / "deepstream.ini"
    config_path.write_text("[property]\n", encoding="utf-8")
    manifest = build_engine_manifest(
        model_id="combat",
        display_name="Combat",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
    )
    backend = DeepStreamDetectionBackend(
        pipeline_config=DeepStreamPipelineConfig(
            device="/dev/video0",
            capture_width=1920,
            capture_height=1080,
            fps=120,
            roi_left=720,
            roi_top=300,
            roi_size=480,
            model_width=256,
            model_height=256,
            nvinfer_config_path=config_path,
        ),
        manifest=manifest,
        roi_width=480,
        roi_height=480,
    )
    import numpy as np

    output = np.zeros((1, 8, 1344), dtype=np.float32)
    output[0, :, 0] = [128, 128, 32, 64, 0.9, 0.1, 0.0, 0.0]

    batch = backend.publish_output_tensor(
        output,
        frame_id=12,
        capture_ts_ns=1000,
        inference_start_ts_ns=1200,
        inference_end_ts_ns=1800,
    )

    assert batch is backend.latest_result()
    assert backend.latest_result(after_frame_id=11) is batch
    assert backend.latest_result(after_frame_id=12) is None
    assert batch.coordinate_space == "roi"
    assert len(batch.detections) == 1
    status = backend.status()
    assert status["selected"] == "deepstream"
    assert status["published_batches"] == 1
    assert status["last_frame_id"] == 12
    assert status["last_error"] == ""


def test_tensor_input_preparer_prefers_gpu_buffer() -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=44,
        metadata={"caps_features": "memory:NVMM", "gst_buffer_pts": 123},
    )
    frame = SimpleNamespace(
        frame_id=7,
        capture_ts_ns=123456789,
        width=320,
        height=320,
        roi_size=320,
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=380,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x640x640")

    prepared = prepare_tensor_input(frame, shape)

    assert prepared.mode == "gpu_buffer"
    assert prepared.buffer is resource.handle
    assert prepared.frame_id == 7
    assert prepared.capture_ts_ns == 123456789
    assert prepared.resource_kind == "gstreamer_sample"
    assert prepared.resource_memory == "nvmm"
    assert prepared.resource_source == "appsink"
    assert prepared.dmabuf_fd == 44
    assert prepared.resource_metadata == {
        "caps_features": "memory:NVMM",
        "gst_buffer_pts": 123,
    }
    assert prepared.resource_width == 320
    assert prepared.resource_height == 320
    assert prepared.resource_pixel_format == "NV12"
    assert prepared.width == 320
    assert prepared.height == 320
    assert prepared.needs_resize is True


def test_gpu_buffer_preprocess_fails_until_real_gpu_ingest_exists() -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
    )
    frame = SimpleNamespace(
        frame_id=7,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=380,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)

    with pytest.raises(TensorPreprocessError, match="nvmm tensor input is not implemented") as exc_info:
        prepare_host_tensor(prepared, shape)
    assert exc_info.value.reason == GPU_RESOURCE_PREPROCESS_NOT_IMPLEMENTED


def test_cpu_image_preprocess_reports_host_tensor_backend() -> None:
    import numpy as np

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    frame = SimpleNamespace(
        frame_id=7,
        width=2,
        height=2,
        source_width=2,
        source_height=2,
        offset_x=0,
        offset_y=0,
        pixel_format="BGR",
        image=image,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x2x2")
    prepared = prepare_tensor_input(frame, shape)

    result = prepare_host_tensor(prepared, shape)

    assert result.backend == "cpu_numpy"
    assert result.zero_copy is False
    assert result.input_mode == "cpu_image"
    assert result.resource_memory == "cpu"
    assert result.tensor.shape == (1, 3, 2, 2)
    assert float(result.tensor[0, 0, 0, 0]) == pytest.approx(1.0)


def test_gpu_resource_preprocessor_can_return_device_tensor_contract() -> None:
    class FakeOwner:
        def release(self) -> None:
            pass

    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
    )
    frame = SimpleNamespace(
        frame_id=7,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=380,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)

    class FakeGpuPreprocessor:
        def prepare(
            self,
            prepared_input: PreparedTensorInput,
            input_shape: TensorInputShape,
        ) -> TensorPreprocessResult:
            return TensorPreprocessResult(
                tensor=DeviceTensor(
                    device_ptr=999,
                    nbytes=1 * 3 * 320 * 320 * 4,
                    shape=(input_shape.batch, input_shape.channels, input_shape.height, input_shape.width),
                    owner=FakeOwner(),
                ),
                backend="fake_nvmm_cuda",
                input_mode=prepared_input.mode,
                resource_kind=prepared_input.resource_kind,
                resource_memory=prepared_input.resource_memory,
                location="device",
                zero_copy=True,
            )

    result = prepare_tensor(prepared, shape, gpu_preprocessor=FakeGpuPreprocessor())

    assert isinstance(result.tensor, DeviceTensor)
    assert result.tensor.device_ptr == 999
    assert result.location == "device"
    assert result.backend == "fake_nvmm_cuda"
    assert result.zero_copy is True
    assert result.debug_payload()["preprocess_location"] == "device"


def test_gpu_resource_preprocessor_can_return_fp16_device_tensor_contract() -> None:
    class FakeOwner:
        def release(self) -> None:
            pass

    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
    )
    frame = SimpleNamespace(
        frame_id=7,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=380,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = TensorInputShape(batch=1, channels=3, height=320, width=320, dtype="float16")
    prepared = prepare_tensor_input(frame, shape)

    class FakeGpuPreprocessor:
        def prepare(
            self,
            prepared_input: PreparedTensorInput,
            input_shape: TensorInputShape,
        ) -> TensorPreprocessResult:
            return TensorPreprocessResult(
                tensor=DeviceTensor(
                    device_ptr=999,
                    nbytes=input_shape.nbytes,
                    shape=(
                        input_shape.batch,
                        input_shape.channels,
                        input_shape.height,
                        input_shape.width,
                    ),
                    dtype="float16",
                    owner=FakeOwner(),
                ),
                backend="fake_nvmm_cuda",
                input_mode=prepared_input.mode,
                resource_kind=prepared_input.resource_kind,
                resource_memory=prepared_input.resource_memory,
                location="device",
                zero_copy=True,
            )

    result = prepare_tensor(prepared, shape, gpu_preprocessor=FakeGpuPreprocessor())

    assert isinstance(result.tensor, DeviceTensor)
    assert result.tensor.dtype == "float16"
    assert result.tensor.nbytes == 1 * 3 * 320 * 320 * 2


def test_gpu_resource_preprocessor_rejects_device_tensor_without_release_owner() -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
    )
    frame = SimpleNamespace(
        frame_id=7,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=380,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)

    class FakeGpuPreprocessor:
        def prepare(
            self,
            prepared_input: PreparedTensorInput,
            input_shape: TensorInputShape,
        ) -> TensorPreprocessResult:
            return TensorPreprocessResult(
                tensor=DeviceTensor(
                    device_ptr=999,
                    nbytes=input_shape.nbytes,
                    shape=(input_shape.batch, input_shape.channels, input_shape.height, input_shape.width),
                ),
                backend="fake_nvmm_cuda",
                input_mode=prepared_input.mode,
                resource_kind=prepared_input.resource_kind,
                resource_memory=prepared_input.resource_memory,
                location="device",
                zero_copy=True,
            )

    with pytest.raises(TensorPreprocessError) as exc_info:
        prepare_tensor(prepared, shape, gpu_preprocessor=FakeGpuPreprocessor())

    assert exc_info.value.reason == "GPU_RESOURCE_PREPROCESS_INVALID"
    assert "owner" in str(exc_info.value)
    assert "release" in str(exc_info.value)


def test_jetson_gpu_resource_preprocessor_fails_closed_until_native_bridge_exists() -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
    )
    frame = SimpleNamespace(
        frame_id=7,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        offset_x=800,
        offset_y=380,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)

    with pytest.raises(TensorPreprocessError) as exc_info:
        prepare_tensor(prepared, shape, gpu_preprocessor=JetsonGpuResourcePreprocessor())

    assert exc_info.value.reason == JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE
    assert "cannot convert nvmm/gstreamer_sample" in str(exc_info.value)


def test_default_jetson_bridge_adapter_reports_native_backend_missing(monkeypatch) -> None:
    import novasight_jetson_preprocess

    novasight_jetson_preprocess._reset_backend_cache()
    monkeypatch.delenv(JETSON_GPU_RESOURCE_BRIDGE_ENV, raising=False)
    monkeypatch.setenv(
        "NOVASIGHT_JETSON_NATIVE_PREPROCESSOR",
        "missing_native_preprocessor_for_test",
    )

    preprocessor = JetsonGpuResourcePreprocessor()
    status = preprocessor.status()

    assert status["module"] == DEFAULT_JETSON_GPU_RESOURCE_BRIDGE_MODULE
    assert status["available"] is False
    assert status["reason"] == JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE
    assert status["abi_version"] == JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
    assert status["abi_compatible"] is True
    assert status["native_ready"] is False
    assert status["native_status"]["reason"] == "native_backend_unavailable"
    assert "missing_native_preprocessor_for_test" in status["detail"]
    novasight_jetson_preprocess._reset_backend_cache()


def test_default_jetson_bridge_adapter_reports_ctypes_backend_without_library(monkeypatch) -> None:
    import novasight_jetson_preprocess

    novasight_jetson_preprocess._reset_backend_cache()
    monkeypatch.delenv(JETSON_GPU_RESOURCE_BRIDGE_ENV, raising=False)
    monkeypatch.delenv("NOVASIGHT_JETSON_NATIVE_PREPROCESSOR", raising=False)

    status = JetsonGpuResourcePreprocessor().status()

    assert status["module"] == DEFAULT_JETSON_GPU_RESOURCE_BRIDGE_MODULE
    assert status["available"] is False
    assert status["reason"] == JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE
    assert status["abi_version"] == JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
    assert status["abi_compatible"] is True
    assert status["native_ready"] is False
    assert status["native_status"]["reason"] == "native_implementation_missing"
    assert "NvBufSurface/EGL/CUDA" in status["detail"]
    assert status["contract"]["abi_version"] == JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
    assert status["contract"]["payload_required_fields"] == list(
        JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_PAYLOAD_FIELDS
    )
    assert status["contract"]["result_required_fields"] == list(
        JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_RESULT_FIELDS
    )
    novasight_jetson_preprocess._reset_backend_cache()


def test_ctypes_native_backend_is_abi_compatible_but_unavailable_without_library() -> None:
    status = JetsonGpuResourcePreprocessor(
        module_name="novasight_jetson_preprocess_native"
    ).status()

    assert status["module"] == "novasight_jetson_preprocess_native"
    assert status["available"] is False
    assert status["reason"] == JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE
    assert status["abi_version"] == JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
    assert status["abi_compatible"] is True
    assert status["native_ready"] is False
    assert status["native_status"]["reason"] == "native_implementation_missing"
    assert status["contract"]["payload_required_fields"] == list(
        JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_PAYLOAD_FIELDS
    )
    assert status["contract"]["result_required_fields"] == list(
        JETSON_GPU_RESOURCE_BRIDGE_REQUIRED_RESULT_FIELDS
    )


def test_ctypes_native_backend_reports_library_status(monkeypatch) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(lambda *args: 0)
            self.novasight_status_json = FakeCFunction(self.status_json)

        def status_json(self, buffer, size):
            del size
            buffer.value = (
                b'{"available":true,"ready":true,"backend":"fake_ctypes",'
                b'"zero_copy":true,"memory_space":"cuda_device"}'
            )
            return 0

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    status = native_backend.status()

    assert status["available"] is True
    assert status["ready"] is True
    assert status["backend"] == "fake_ctypes"
    assert status["library"] == "/tmp/fake-novasight-jetson.so"
    assert status["required_abi_version"] == native_backend.ABI_VERSION
    assert status["abi_version"] == native_backend.ABI_VERSION
    assert status["abi_compatible"] is True
    assert status["zero_copy"] is True
    assert status["memory_space"] == "cuda_device"
    assert status["contract"]["ctypes_prepare_abi"].startswith("int prepare")
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_ready_status_without_output_contract(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(lambda *args: 0)
            self.novasight_status_json = FakeCFunction(self.status_json)

        def status_json(self, buffer, size):
            del size
            buffer.value = b'{"available":true,"ready":true,"backend":"fake_ctypes"}'
            return 0

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    status = native_backend.status()

    assert status["available"] is False
    assert status["ready"] is False
    assert status["reason"] == "native_output_contract_invalid"
    assert "zero_copy=true" in status["detail"]
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_library_without_status_symbol(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(lambda *args: 0)

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    status = native_backend.status()

    assert status["available"] is False
    assert status["ready"] is False
    assert status["reason"] == "native_readiness_status_required"
    assert "zero_copy=true" in status["detail"]
    native_backend._reset_library_cache()


def test_ctypes_native_backend_reports_status_hook_failure(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(lambda *args: 0)
            self.novasight_status_json = FakeCFunction(self.status_json)

        def status_json(self, buffer, size):
            del size
            buffer.value = b'{"reason":"cuda_context_unavailable"}'
            return 7

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    status = native_backend.status()

    assert status["available"] is False
    assert status["ready"] is False
    assert status["reason"] == "native_status_failed"
    assert "cuda_context_unavailable" in status["detail"]

    with pytest.raises(RuntimeError, match="native status failed"):
        native_backend.prepare_tensor(_ctypes_native_payload())
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_prepare_without_status_symbol(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(lambda *args: 0)

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    with pytest.raises(RuntimeError, match="must export .*status"):
        native_backend.prepare_tensor(_ctypes_native_payload())
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_library_without_c_abi_version(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_prepare_tensor_json = FakeCFunction(lambda *args: 0)

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    status = native_backend.status()

    assert status["available"] is False
    assert status["reason"] == "native_abi_version_missing"
    assert status["required_abi_version"] == native_backend.ABI_VERSION
    assert status["abi_version"] is None
    assert status["abi_compatible"] is False
    assert "must export" in status["detail"]
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_incompatible_c_abi_version(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: 999)
            self.novasight_prepare_tensor_json = FakeCFunction(lambda *args: 0)

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    status = native_backend.status()

    assert status["available"] is False
    assert status["reason"] == "native_abi_incompatible"
    assert status["required_abi_version"] == native_backend.ABI_VERSION
    assert status["abi_version"] == 999
    assert status["abi_compatible"] is False
    assert "ABI mismatch" in status["detail"]
    native_backend._reset_library_cache()


def test_ctypes_native_backend_prepares_dmabuf_payload_and_releases_owner(
    monkeypatch,
) -> None:
    import json

    import novasight_jetson_preprocess_native as native_backend

    captured_payloads: list[dict] = []
    released: list[int] = []

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(self.prepare_json)
            self.novasight_release_tensor = FakeCFunction(self.release_tensor)
            self.novasight_status_json = FakeCFunction(self.status_json)

        def status_json(self, buffer, size):
            del size
            buffer.value = (
                b'{"available":true,"ready":true,"backend":"fake_ctypes",'
                b'"zero_copy":true,"memory_space":"cuda_device"}'
            )
            return 0

        def prepare_json(self, payload_ptr, buffer, size):
            del size
            payload = json.loads(payload_ptr.value.decode("utf-8"))
            captured_payloads.append(payload)
            buffer.value = (
                b'{"device_ptr":123456,"nbytes":1228800,'
                b'"shape":[1,3,320,320],"zero_copy":true,'
                b'"memory_space":"cuda_device","release_token":77}'
            )
            return 0

        def release_tensor(self, token):
            released.append(int(token))
            return 0

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    result = native_backend.prepare_tensor(
        _ctypes_native_payload(
            capture_ts_ns=1234567890,
            resource_metadata={"memory_types": ["DMABuf"], "opaque": object()},
        )
    )

    assert captured_payloads
    assert "resource_handle" not in captured_payloads[0]
    assert captured_payloads[0]["frame_id"] == 17
    assert captured_payloads[0]["capture_ts_ns"] == 1234567890
    assert captured_payloads[0]["dmabuf_fd"] == 42
    assert captured_payloads[0]["resource_width"] == 320
    assert captured_payloads[0]["resource_height"] == 320
    assert captured_payloads[0]["resource_pixel_format"] == "NV12"
    assert captured_payloads[0]["resource_metadata"]["opaque"].startswith("<object")
    assert result["device_ptr"] == 123456
    assert result["nbytes"] == 1228800
    assert result["backend"] == "novasight_jetson_preprocess_native:ctypes"
    result["owner"].release()
    assert released == [77]
    native_backend._reset_library_cache()


def test_ctypes_native_backend_owner_release_fails_on_native_error(
    monkeypatch,
) -> None:
    import pytest

    import novasight_jetson_preprocess_native as native_backend

    released: list[int] = []

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(self.prepare_json)
            self.novasight_release_tensor = FakeCFunction(self.release_tensor)
            self.novasight_status_json = FakeCFunction(self.status_json)

        def status_json(self, buffer, size):
            del size
            buffer.value = (
                b'{"available":true,"ready":true,"backend":"fake_ctypes",'
                b'"zero_copy":true,"memory_space":"cuda_device"}'
            )
            return 0

        def prepare_json(self, payload_ptr, buffer, size):
            del payload_ptr, size
            buffer.value = (
                b'{"device_ptr":123456,"nbytes":1228800,'
                b'"shape":[1,3,320,320],"zero_copy":true,'
                b'"memory_space":"cuda_device","release_token":77}'
            )
            return 0

        def release_tensor(self, token):
            released.append(int(token))
            return 9

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    result = native_backend.prepare_tensor(
        _ctypes_native_payload(capture_ts_ns=1234567890)
    )

    with pytest.raises(RuntimeError, match="native release failed.*token=77.*rc=9"):
        result["owner"].release()
    assert released == [77]
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_release_token_without_release_symbol(
    monkeypatch,
) -> None:
    import pytest

    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(self.prepare_json)
            self.novasight_status_json = FakeCFunction(self.status_json)

        def status_json(self, buffer, size):
            del size
            buffer.value = (
                b'{"available":true,"ready":true,"backend":"fake_ctypes",'
                b'"zero_copy":true,"memory_space":"cuda_device"}'
            )
            return 0

        def prepare_json(self, payload_ptr, buffer, size):
            del payload_ptr, size
            buffer.value = (
                b'{"device_ptr":123456,"nbytes":1228800,'
                b'"shape":[1,3,320,320],"zero_copy":true,'
                b'"memory_space":"cuda_device","release_token":77}'
            )
            return 0

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    with pytest.raises(RuntimeError, match="release_token.*novasight_release_tensor"):
        native_backend.prepare_tensor(_ctypes_native_payload(capture_ts_ns=1234567890))
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_result_without_release_token(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(self.prepare_json)
            self.novasight_status_json = FakeCFunction(self.status_json)

        def status_json(self, buffer, size):
            del size
            buffer.value = (
                b'{"available":true,"ready":true,"backend":"fake_ctypes",'
                b'"zero_copy":true,"memory_space":"cuda_device"}'
            )
            return 0

        def prepare_json(self, payload_ptr, buffer, size):
            del payload_ptr, size
            buffer.value = (
                b'{"device_ptr":123456,"nbytes":1228800,'
                b'"shape":[1,3,320,320],"zero_copy":true,'
                b'"memory_space":"cuda_device"}'
            )
            return 0

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    with pytest.raises(RuntimeError, match="release_token.*positive integer"):
        native_backend.prepare_tensor(_ctypes_native_payload(capture_ts_ns=1234567890))
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_result_without_device_contract(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(self.prepare_json)
            self.novasight_status_json = FakeCFunction(self.status_json)
            self.novasight_release_tensor = FakeCFunction(lambda token: 0)

        def status_json(self, buffer, size):
            del size
            buffer.value = (
                b'{"available":true,"ready":true,"backend":"fake_ctypes",'
                b'"zero_copy":true,"memory_space":"cuda_device"}'
            )
            return 0

        def prepare_json(self, payload_ptr, buffer, size):
            del payload_ptr, size
            buffer.value = b'{"device_ptr":123456,"nbytes":1228800,"release_token":77}'
            return 0

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    with pytest.raises(RuntimeError, match="zero_copy=true"):
        native_backend.prepare_tensor(_ctypes_native_payload())
    native_backend._reset_library_cache()


def test_direct_ctypes_native_bridge_accepts_nvmm_resource_with_dmabuf(
    monkeypatch,
) -> None:
    import json

    import novasight_jetson_preprocess_native as native_backend

    captured_payloads: list[dict] = []

    class FakeCFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class FakeLibrary:
        def __init__(self) -> None:
            self.novasight_abi_version = FakeCFunction(lambda: native_backend.ABI_VERSION)
            self.novasight_prepare_tensor_json = FakeCFunction(self.prepare_json)
            self.novasight_status_json = FakeCFunction(self.status_json)
            self.novasight_release_tensor = FakeCFunction(lambda token: 0)

        def status_json(self, buffer, size):
            del size
            buffer.value = (
                b'{"available":true,"ready":true,"backend":"direct_ctypes",'
                b'"zero_copy":true,"memory_space":"cuda_device"}'
            )
            return 0

        def prepare_json(self, payload_ptr, buffer, size):
            del size
            payload = json.loads(payload_ptr.value.decode("utf-8"))
            captured_payloads.append(payload)
            buffer.value = (
                b'{"device_ptr":456789,"nbytes":1228800,'
                b'"shape":[1,3,320,320],"zero_copy":true,'
                b'"memory_space":"cuda_device","backend":"direct_ctypes",'
                b'"release_token":77}'
            )
            return 0

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=91,
        metadata={"caps_features": "memory:NVMM", "memory_types": ["NVMM"]},
    )
    frame = SimpleNamespace(
        frame_id=13,
        capture_ts_ns=1_234_567_890,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)
    preprocessor = JetsonGpuResourcePreprocessor(
        module_name="novasight_jetson_preprocess_native"
    )

    result = prepare_tensor(prepared, shape, gpu_preprocessor=preprocessor)

    assert captured_payloads
    assert captured_payloads[0]["resource_memory"] == "nvmm"
    assert captured_payloads[0]["resource_source"] == "appsink"
    assert captured_payloads[0]["dmabuf_fd"] == 91
    assert captured_payloads[0]["resource_width"] == 320
    assert captured_payloads[0]["resource_height"] == 320
    assert captured_payloads[0]["resource_pixel_format"] == "NV12"
    assert isinstance(result.tensor, DeviceTensor)
    assert result.tensor.device_ptr == 456789
    assert result.backend == "direct_ctypes"
    status = preprocessor.status()
    assert status["capabilities"]["memory"] == ["nvmm", "dmabuf"]
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_missing_dmabuf_fd(monkeypatch) -> None:
    import novasight_jetson_preprocess_native as native_backend

    class FakeCFunction:
        def __init__(self):
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            raise AssertionError("native prepare must not be called without dmabuf_fd")

    class FakeLibrary:
        novasight_prepare_tensor_json = FakeCFunction()

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(native_backend.ctypes, "CDLL", lambda path: FakeLibrary())

    with pytest.raises(RuntimeError, match="requires dmabuf_fd"):
        native_backend.prepare_tensor(
            _ctypes_native_payload(resource_memory="nvmm", dmabuf_fd=None)
        )
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_missing_frame_timestamp_before_native_call(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(
        native_backend.ctypes,
        "CDLL",
        lambda path: pytest.fail("native library must not be loaded for bad payload"),
    )
    payload = _ctypes_native_payload()
    payload.pop("capture_ts_ns")

    with pytest.raises(RuntimeError, match="capture_ts_ns"):
        native_backend.prepare_tensor(payload)
    native_backend._reset_library_cache()


def test_ctypes_native_backend_rejects_non_appsink_source_before_native_call(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess_native as native_backend

    native_backend._reset_library_cache()
    monkeypatch.setenv(native_backend.LIBRARY_ENV, "/tmp/fake-novasight-jetson.so")
    monkeypatch.setattr(
        native_backend.ctypes,
        "CDLL",
        lambda path: pytest.fail("native library must not be loaded for bad payload"),
    )

    with pytest.raises(RuntimeError, match="resource_source=appsink"):
        native_backend.prepare_tensor(_ctypes_native_payload(resource_source="synthetic"))
    native_backend._reset_library_cache()


def test_default_jetson_bridge_adapter_reloads_when_native_backend_env_changes(
    monkeypatch,
) -> None:
    import novasight_jetson_preprocess

    novasight_jetson_preprocess._reset_backend_cache()
    monkeypatch.delenv(JETSON_GPU_RESOURCE_BRIDGE_ENV, raising=False)
    monkeypatch.setenv(
        "NOVASIGHT_JETSON_NATIVE_PREPROCESSOR",
        "missing_reload_native_preprocessor_for_test",
    )

    missing_status = JetsonGpuResourcePreprocessor().status()
    assert missing_status["available"] is False
    assert "missing_reload_native_preprocessor_for_test" in missing_status["detail"]

    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=91,
    )
    frame = SimpleNamespace(
        frame_id=12,
        capture_ts_ns=1_234_567_890,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)
    releases: list[str] = []

    class FakeOwner:
        def release(self) -> None:
            releases.append("released")

    backend = SimpleNamespace(
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "fake_reload_backend",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_tensor=lambda payload: {
            "device_ptr": 987,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "shape": (1, 3, 320, 320),
            "backend": "fake_reload_backend",
            "zero_copy": True,
            "memory_space": "cuda_device",
            "owner": FakeOwner(),
        },
    )
    monkeypatch.setitem(sys.modules, "reload_native_preprocessor_for_test", backend)
    monkeypatch.setenv(
        "NOVASIGHT_JETSON_NATIVE_PREPROCESSOR",
        "reload_native_preprocessor_for_test",
    )

    preprocessor = JetsonGpuResourcePreprocessor()
    status = preprocessor.status()
    result = prepare_tensor(prepared, shape, gpu_preprocessor=preprocessor)

    assert status["available"] is True
    assert status["native_ready"] is True
    assert status["native_status"]["backend"] == "fake_reload_backend"
    assert isinstance(result.tensor, DeviceTensor)
    assert result.tensor.device_ptr == 987
    assert result.backend == "fake_reload_backend"
    result.tensor.owner.release()
    assert releases == ["released"]
    novasight_jetson_preprocess._reset_backend_cache()


def test_jetson_gpu_resource_preprocessor_uses_native_bridge_when_available(monkeypatch) -> None:
    resource_handle = object()
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=resource_handle,
        memory="nvmm",
        width=640,
        height=640,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=91,
        metadata={
            "caps_features": "memory:NVMM",
            "memory_types": ["NVMM"],
            "gst_buffer_pts": 987654,
        },
    )
    frame = SimpleNamespace(
        frame_id=9,
        capture_ts_ns=9876543210,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_x=640,
        roi_y=220,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)
    calls: list[dict] = []
    releases: list[str] = []

    class FakeOwner:
        def release(self) -> None:
            releases.append("released")

    def native_prepare(payload: dict) -> dict:
        calls.append(payload)
        return {
            "device_ptr": 123456,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "shape": (1, 3, 320, 320),
            "dtype": "float32",
            "backend": "fake_jetson_native",
            "zero_copy": True,
            "memory_space": "cuda_device",
            "owner": FakeOwner(),
        }

    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={"memory": ["nvmm", "dmabuf"], "formats": ["NV12"]},
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "fake_jetson_native",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_tensor=native_prepare,
    )
    monkeypatch.setitem(sys.modules, "fake_jetson_native_bridge", module)

    preprocessor = JetsonGpuResourcePreprocessor(module_name="fake_jetson_native_bridge")
    result = prepare_tensor(prepared, shape, gpu_preprocessor=preprocessor)

    assert calls
    assert calls[0]["resource_handle"] is resource_handle
    assert calls[0]["frame_id"] == 9
    assert calls[0]["capture_ts_ns"] == 9876543210
    assert calls[0]["resource_memory"] == "nvmm"
    assert calls[0]["resource_kind"] == "gstreamer_sample"
    assert calls[0]["resource_source"] == "appsink"
    assert calls[0]["dmabuf_fd"] == 91
    assert calls[0]["resource_metadata"] == {
        "caps_features": "memory:NVMM",
        "memory_types": ["NVMM"],
        "gst_buffer_pts": 987654,
    }
    assert calls[0]["width"] == 640
    assert calls[0]["height"] == 640
    assert calls[0]["source_width"] == 1920
    assert calls[0]["source_height"] == 1080
    assert calls[0]["roi_offset_x"] == 640
    assert calls[0]["roi_offset_y"] == 220
    assert calls[0]["needs_resize"] is True
    assert calls[0]["model_shape"] == {
        "batch": 1,
        "channels": 3,
        "height": 320,
        "width": 320,
    }
    assert isinstance(result.tensor, DeviceTensor)
    assert result.tensor.device_ptr == 123456
    assert result.tensor.nbytes == 1 * 3 * 320 * 320 * 4
    assert result.location == "device"
    assert result.backend == "fake_jetson_native"
    assert result.zero_copy is True
    result.tensor.owner.release()
    assert releases == ["released"]
    status = preprocessor.status()
    assert status["available"] is True
    assert status["required_abi_version"] == JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
    assert status["abi_version"] == JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
    assert status["abi_compatible"] is True
    assert status["capabilities"] == {"memory": ["nvmm", "dmabuf"], "formats": ["NV12"]}


def test_jetson_gpu_resource_preprocessor_passes_fp16_dtype_to_native_bridge(
    monkeypatch,
) -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=91,
    )
    frame = SimpleNamespace(
        frame_id=9,
        capture_ts_ns=1_234_567_890,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = TensorInputShape(batch=1, channels=3, height=320, width=320, dtype="fp16")
    prepared = prepare_tensor_input(frame, shape)
    calls: list[dict] = []
    releases: list[str] = []

    class FakeOwner:
        def release(self) -> None:
            releases.append("released")

    def native_prepare(payload: dict) -> dict:
        calls.append(payload)
        return {
            "device_ptr": 123456,
            "nbytes": shape.nbytes,
            "shape": (1, 3, 320, 320),
            "dtype": "float16",
            "backend": "fake_jetson_native_fp16",
            "zero_copy": True,
            "memory_space": "cuda_device",
            "owner": FakeOwner(),
        }

    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={
            "memory": ["nvmm", "dmabuf"],
            "formats": ["NV12"],
            "dtypes": ["float16"],
        },
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "fake_jetson_native_fp16",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_tensor=native_prepare,
    )
    monkeypatch.setitem(sys.modules, "fake_jetson_native_bridge_fp16", module)

    preprocessor = JetsonGpuResourcePreprocessor(
        module_name="fake_jetson_native_bridge_fp16"
    )
    result = prepare_tensor(prepared, shape, gpu_preprocessor=preprocessor)

    assert calls
    assert calls[0]["dtype"] == "float16"
    assert isinstance(result.tensor, DeviceTensor)
    assert result.tensor.dtype == "float16"
    assert result.tensor.nbytes == 1 * 3 * 320 * 320 * 2
    result.tensor.owner.release()
    assert releases == ["released"]


def test_jetson_gpu_resource_preprocessor_rejects_result_without_device_contract(
    monkeypatch,
) -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=91,
    )
    frame = SimpleNamespace(
        frame_id=10,
        capture_ts_ns=1_234_567_890,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)
    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={"memory": ["nvmm"], "formats": ["NV12"]},
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "missing_result_contract_bridge",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_tensor=lambda payload: {
            "device_ptr": 123456,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "shape": (1, 3, 320, 320),
        },
    )
    monkeypatch.setitem(sys.modules, "missing_result_contract_bridge", module)

    with pytest.raises(TensorPreprocessError) as exc_info:
        prepare_tensor(
            prepared,
            shape,
            gpu_preprocessor=JetsonGpuResourcePreprocessor(
                module_name="missing_result_contract_bridge"
            ),
        )

    assert exc_info.value.reason == JETSON_GPU_RESOURCE_BRIDGE_INVALID
    assert "zero_copy=true" in str(exc_info.value)


def test_jetson_gpu_resource_preprocessor_requires_explicit_ready_status(
    monkeypatch,
) -> None:
    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={"memory": ["nvmm"], "formats": ["NV12"]},
        prepare_tensor=lambda payload: {
            "device_ptr": 123456,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "shape": (1, 3, 320, 320),
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
    )
    monkeypatch.setitem(sys.modules, "implicit_ready_jetson_bridge", module)

    status = JetsonGpuResourcePreprocessor(
        module_name="implicit_ready_jetson_bridge"
    ).status()

    assert status["available"] is False
    assert status["native_ready"] is False
    assert status["reason"] == JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE
    assert status["native_status"]["reason"] == "native_readiness_status_required"


def test_jetson_gpu_resource_preprocessor_rejects_ready_status_without_output_contract(
    monkeypatch,
) -> None:
    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={"memory": ["nvmm"], "formats": ["NV12"]},
        status=lambda: {"available": True, "ready": True, "backend": "not_enough"},
        prepare_tensor=lambda payload: {
            "device_ptr": 123456,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "shape": (1, 3, 320, 320),
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
    )
    monkeypatch.setitem(sys.modules, "ready_without_contract_bridge", module)

    status = JetsonGpuResourcePreprocessor(
        module_name="ready_without_contract_bridge"
    ).status()

    assert status["available"] is False
    assert status["native_ready"] is False
    assert status["native_status"]["reason"] == "native_output_contract_invalid"
    assert "zero_copy=true" in status["detail"]


def test_jetson_gpu_resource_preprocessor_rejects_incompatible_native_bridge(
    monkeypatch,
) -> None:
    module = SimpleNamespace(
        ABI_VERSION=999,
        prepare_tensor=lambda payload: {
            "device_ptr": 1,
            "nbytes": 1 * 3 * 320 * 320 * 4,
        },
    )
    monkeypatch.setitem(sys.modules, "incompatible_jetson_native_bridge", module)

    preprocessor = JetsonGpuResourcePreprocessor(
        module_name="incompatible_jetson_native_bridge"
    )
    status = preprocessor.status()

    assert status["available"] is False
    assert status["reason"] == JETSON_GPU_RESOURCE_BRIDGE_INVALID
    assert status["required_abi_version"] == JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION
    assert status["abi_version"] == 999
    assert status["abi_compatible"] is False
    assert "required ABI_VERSION" in status["detail"]


def test_jetson_gpu_resource_preprocessor_rejects_unsupported_payload_before_native_call(
    monkeypatch,
) -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=640,
        height=640,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=91,
    )
    frame = SimpleNamespace(
        frame_id=11,
        capture_ts_ns=1_234_567_890,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_x=640,
        roi_y=220,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)
    calls: list[dict] = []

    def native_prepare(payload: dict) -> dict:
        calls.append(payload)
        return {
            "device_ptr": 1,
            "nbytes": 1 * 3 * 320 * 320 * 4,
        }

    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={"memory": ["dmabuf"], "formats": ["RGBA"]},
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "unsupported_payload_jetson_bridge",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_tensor=native_prepare,
    )
    monkeypatch.setitem(sys.modules, "unsupported_payload_jetson_bridge", module)

    with pytest.raises(TensorPreprocessError) as exc_info:
        prepare_tensor(
            prepared,
            shape,
            gpu_preprocessor=JetsonGpuResourcePreprocessor(
                module_name="unsupported_payload_jetson_bridge"
            ),
        )

    assert exc_info.value.reason == JETSON_GPU_RESOURCE_BRIDGE_INVALID
    assert "does not support resource_memory='nvmm'" in str(exc_info.value)
    assert calls == []


def test_jetson_gpu_resource_preprocessor_rejects_unsupported_resource_source_before_native_call(
    monkeypatch,
) -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=640,
        height=640,
        pixel_format="NV12",
        source="synthetic",
        dmabuf_fd=91,
    )
    frame = SimpleNamespace(
        frame_id=11,
        capture_ts_ns=1_234_567_890,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)
    calls: list[dict] = []

    def native_prepare(payload: dict) -> dict:
        calls.append(payload)
        return {
            "device_ptr": 1,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "zero_copy": True,
            "memory_space": "cuda_device",
        }

    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={
            "memory": ["nvmm"],
            "resource_source": ["appsink"],
            "formats": ["NV12"],
        },
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "unsupported_source_jetson_bridge",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_tensor=native_prepare,
    )
    monkeypatch.setitem(sys.modules, "unsupported_source_jetson_bridge", module)

    with pytest.raises(TensorPreprocessError) as exc_info:
        prepare_tensor(
            prepared,
            shape,
            gpu_preprocessor=JetsonGpuResourcePreprocessor(
                module_name="unsupported_source_jetson_bridge"
            ),
        )

    assert exc_info.value.reason == JETSON_GPU_RESOURCE_BRIDGE_INVALID
    assert "requires resource_source='appsink'" in str(exc_info.value)
    assert calls == []


def test_jetson_gpu_resource_preprocessor_rejects_missing_frame_timestamp_before_native_call(
    monkeypatch,
) -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=91,
    )
    frame = SimpleNamespace(
        frame_id=12,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)
    calls: list[dict] = []
    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={"memory": ["nvmm"], "resource_source": ["appsink"], "formats": ["NV12"]},
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "missing_frame_timestamp_bridge",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_tensor=lambda payload: calls.append(payload),
    )
    monkeypatch.setitem(sys.modules, "missing_frame_timestamp_bridge", module)

    with pytest.raises(TensorPreprocessError) as exc_info:
        prepare_tensor(
            prepared,
            shape,
            gpu_preprocessor=JetsonGpuResourcePreprocessor(
                module_name="missing_frame_timestamp_bridge"
            ),
        )

    assert exc_info.value.reason == JETSON_GPU_RESOURCE_BRIDGE_INVALID
    assert "capture_ts_ns" in str(exc_info.value)
    assert calls == []


def test_jetson_gpu_resource_preprocessor_rejects_missing_dmabuf_before_native_call(
    monkeypatch,
) -> None:
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
        dmabuf_fd=None,
    )
    frame = SimpleNamespace(
        frame_id=12,
        capture_ts_ns=1_234_567_890,
        width=320,
        height=320,
        source_width=1920,
        source_height=1080,
        image=None,
        frame_resource=resource,
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x320x320")
    prepared = prepare_tensor_input(frame, shape)
    calls: list[dict] = []
    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={"memory": ["nvmm"], "resource_source": ["appsink"], "formats": ["NV12"]},
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "missing_dmabuf_bridge",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_tensor=lambda payload: calls.append(payload),
    )
    monkeypatch.setitem(sys.modules, "missing_dmabuf_bridge", module)

    with pytest.raises(TensorPreprocessError) as exc_info:
        prepare_tensor(
            prepared,
            shape,
            gpu_preprocessor=JetsonGpuResourcePreprocessor(module_name="missing_dmabuf_bridge"),
        )

    assert exc_info.value.reason == JETSON_GPU_RESOURCE_BRIDGE_INVALID
    assert "dmabuf_fd" in str(exc_info.value)
    assert calls == []


def test_jetson_gpu_resource_preprocessor_rejects_unready_native_bridge(monkeypatch) -> None:
    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        CAPABILITIES={"memory": ["nvmm"], "formats": ["NV12"]},
        status=lambda: {
            "available": False,
            "reason": "cuda_context_unavailable",
            "detail": "CUDA context is not initialized",
        },
        prepare_tensor=lambda payload: {
            "device_ptr": 1,
            "nbytes": 1 * 3 * 320 * 320 * 4,
        },
    )
    monkeypatch.setitem(sys.modules, "unready_jetson_native_bridge", module)

    preprocessor = JetsonGpuResourcePreprocessor(module_name="unready_jetson_native_bridge")
    status = preprocessor.status()

    assert status["available"] is False
    assert status["reason"] == JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE
    assert status["abi_compatible"] is True
    assert status["native_ready"] is False
    assert status["native_status"]["reason"] == "cuda_context_unavailable"
    assert "CUDA context is not initialized" in status["detail"]


def test_jetson_gpu_resource_preprocessor_status_uses_env_override(monkeypatch) -> None:
    module = SimpleNamespace(
        ABI_VERSION=JETSON_GPU_RESOURCE_BRIDGE_ABI_VERSION,
        status=lambda: {
            "available": True,
            "ready": True,
            "backend": "env_jetson_native_bridge",
            "zero_copy": True,
            "memory_space": "cuda_device",
        },
        prepare_nvmm_tensor=lambda payload: {
            "device_ptr": 1,
            "nbytes": 1 * 3 * 320 * 320 * 4,
            "zero_copy": True,
            "memory_space": "cuda_device",
        }
    )
    monkeypatch.setitem(sys.modules, "env_jetson_native_bridge", module)
    monkeypatch.setenv(JETSON_GPU_RESOURCE_BRIDGE_ENV, "env_jetson_native_bridge")

    preprocessor = JetsonGpuResourcePreprocessor()
    status = preprocessor.status()

    assert status["available"] is True
    assert status["module"] == "env_jetson_native_bridge"
    assert preprocessor.module_name == "env_jetson_native_bridge"
    monkeypatch.delenv(JETSON_GPU_RESOURCE_BRIDGE_ENV)
    assert preprocessor.module_name != DEFAULT_JETSON_GPU_RESOURCE_BRIDGE_MODULE


def test_inference_runtime_injects_gpu_preprocessor_into_tensorrt_candidates() -> None:
    class FakeCurrentEngine:
        engine_id = "fake"

        def __init__(self) -> None:
            self.gpu_preprocessor = None

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"selected": self.engine_id, "available": True, "loaded": False}

        def set_gpu_preprocessor(self, value) -> None:
            self.gpu_preprocessor = value

    current = FakeCurrentEngine()
    preprocessor = JetsonGpuResourcePreprocessor()
    runtime = InferenceRuntime(current)

    runtime.configure(gpu_preprocessor=preprocessor)
    candidate = runtime._engine_for_artifact(Path("model.engine"))

    assert current.gpu_preprocessor is preprocessor
    assert isinstance(candidate, TensorRtInferenceEngine)
    assert candidate._gpu_preprocessor is preprocessor


def test_tensorrt_device_tensor_execution_binds_external_input_without_h2d() -> None:
    import numpy as np

    class FakeMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    class FakeCudaRuntime:
        cudaMemcpyKind = FakeMemcpyKind

        def __init__(self) -> None:
            self.memcpy_kinds: list[int] = []

        def cudaMemcpyAsync(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> int:
            self.memcpy_kinds.append(int(kind))
            return 0

        def cudaStreamSynchronize(self, stream: int) -> int:
            return 0

    class FakeContext:
        def __init__(self) -> None:
            self.input_bindings: list[int] = []
            self.executed_streams: list[int] = []

        def set_tensor_address(self, name: str, pointer: int) -> None:
            if name == "images":
                self.input_bindings.append(int(pointer))

        def execute_async_v3(self, stream: int) -> bool:
            self.executed_streams.append(int(stream))
            return True

    cudart = FakeCudaRuntime()
    context = FakeContext()
    released: list[str] = []

    class FakeOwner:
        def release(self) -> None:
            released.append("released")

    engine = TensorRtInferenceEngine()
    engine._cudart = cudart
    engine._context = context
    engine._host_input = np.empty((1, 3, 2, 2), dtype=np.float32)
    engine._device_input = 111
    engine._stream = 77
    engine._input_name = "images"
    engine._output_name = "output0"
    engine._output_shape = (1, 6, 1)
    engine._host_outputs = {"output0": np.zeros(6, dtype=np.float32)}
    engine._device_outputs = {"output0": 222}
    engine._classes = ["target"]

    detections, debug = engine._execute(
        DeviceTensor(
            device_ptr=999,
            nbytes=engine._host_input.nbytes,
            shape=tuple(engine._host_input.shape),
            owner=FakeOwner(),
        )
    )

    assert detections == []
    assert context.input_bindings == [999, 111]
    assert released == ["released"]
    assert context.executed_streams == [77]
    assert cudart.memcpy_kinds == [FakeMemcpyKind.cudaMemcpyDeviceToHost]
    assert debug["timings"]["input_location"] == "device"
    assert debug["timings"]["h2d_enqueue_ms"] == 0.0
    assert debug["timings"]["device_bind_ms"] >= 0.0
    assert debug["timings"]["device_owner_release"] == "released"


def test_tensorrt_device_tensor_execution_accepts_fp16_input_without_h2d() -> None:
    import numpy as np

    class FakeMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    class FakeCudaRuntime:
        cudaMemcpyKind = FakeMemcpyKind

        def __init__(self) -> None:
            self.memcpy_kinds: list[int] = []

        def cudaMemcpyAsync(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> int:
            self.memcpy_kinds.append(int(kind))
            return 0

        def cudaStreamSynchronize(self, stream: int) -> int:
            return 0

    class FakeContext:
        def __init__(self) -> None:
            self.input_bindings: list[int] = []

        def set_tensor_address(self, name: str, pointer: int) -> None:
            if name == "images":
                self.input_bindings.append(int(pointer))

        def execute_async_v3(self, stream: int) -> bool:
            return True

    cudart = FakeCudaRuntime()
    context = FakeContext()
    engine = TensorRtInferenceEngine()
    engine._cudart = cudart
    engine._context = context
    engine._host_input = np.empty((1, 3, 2, 2), dtype=np.float16)
    engine._device_input = 111
    engine._stream = 77
    engine._input_name = "images"
    engine._output_name = "output0"
    engine._output_shape = (1, 6, 1)
    engine._host_outputs = {"output0": np.zeros(6, dtype=np.float32)}
    engine._device_outputs = {"output0": 222}
    engine._classes = ["target"]

    detections, debug = engine._execute(
        DeviceTensor(
            device_ptr=999,
            nbytes=engine._host_input.nbytes,
            shape=tuple(engine._host_input.shape),
            dtype="float16",
        )
    )

    assert detections == []
    assert context.input_bindings == [999, 111]
    assert cudart.memcpy_kinds == [FakeMemcpyKind.cudaMemcpyDeviceToHost]
    assert debug["timings"]["input_location"] == "device"
    assert debug["timings"]["h2d_enqueue_ms"] == 0.0


def test_tensor_input_preparer_uses_cpu_image_fallback() -> None:
    frame = SimpleNamespace(
        frame_id=7,
        width=640,
        height=640,
        roi_size=640,
        source_width=1920,
        source_height=1080,
        offset_x=640,
        offset_y=220,
        image=object(),
        gpu_buffer=None,
    )
    shape = parse_tensor_input_shape("1x3x640x640")

    prepared = prepare_tensor_input(frame, shape)

    assert prepared.mode == "cpu_image"
    assert prepared.resource_kind == "cpu_image"
    assert prepared.resource_memory == "cpu"
    assert prepared.needs_resize is False


def test_tensor_input_preparer_reads_captured_frame_roi_contract() -> None:
    frame = CapturedFrame(
        frame_id=7,
        width=640,
        height=640,
        pixel_format="BGR",
        ts_ns=123,
        capture_wait_ms=0.1,
        image=object(),
        source_width=1920,
        source_height=1080,
        roi_size=640,
        roi_offset_x=640,
        roi_offset_y=220,
    )
    shape = parse_tensor_input_shape("1x3x320x320")

    prepared = prepare_tensor_input(frame, shape)

    assert prepared.width == 640
    assert prepared.height == 640
    assert prepared.frame_id == 7
    assert prepared.capture_ts_ns == 123
    assert prepared.source_width == 1920
    assert prepared.source_height == 1080
    assert prepared.offset_x == 640
    assert prepared.offset_y == 220
    assert prepared.needs_resize is True


def test_model_detections_are_normalized_to_roi_coordinates() -> None:
    prepared = PreparedTensorInput(
        mode="cpu_image",
        buffer=object(),
        frame_id=1,
        capture_ts_ns=123,
        width=960,
        height=960,
        pixel_format="BGR",
        source_width=1920,
        source_height=1080,
        offset_x=480,
        offset_y=60,
        needs_resize=True,
    )
    shape = TensorInputShape(batch=1, channels=3, height=320, width=320)
    detections = [
        InferenceDetection.from_xyxy(
            cls_id=0,
            score=0.9,
            x1=150,
            y1=150,
            x2=170,
            y2=190,
        )
    ]

    scaled = map_model_detections_to_roi_frame(
        detections,
        prepared=prepared,
        shape=shape,
    )
    preprocess_debug = _preprocess_debug(prepared, shape)

    assert preprocess_debug["resource_kind"] == ""
    assert preprocess_debug["resource_memory"] == ""
    assert scaled[0].x1 == pytest.approx(450.0)
    assert scaled[0].y1 == pytest.approx(450.0)
    assert scaled[0].x2 == pytest.approx(510.0)
    assert scaled[0].y2 == pytest.approx(570.0)
    assert preprocess_debug["coordinate_transform"] == {
        "model_width": 320,
        "model_height": 320,
        "roi_x": 480,
        "roi_y": 60,
        "roi_width": 960,
        "roi_height": 960,
        "capture_width": 1920,
        "capture_height": 1080,
    }


def test_runtime_falls_back_to_unavailable_engine() -> None:
    class FakeUnavailableEngine:
        engine_id = "fake"

        def available(self) -> bool:
            return False

        def last_reason(self) -> str:
            return "fake unavailable"

        def status(self) -> dict:
            raise AssertionError("runtime should not call status while falling back")

        def load(self, artifact_path, classes, input_shape) -> None:
            raise AssertionError("not used")

        def infer(self, frame) -> InferenceResult:
            raise AssertionError("not used")

    runtime = InferenceRuntime(FakeUnavailableEngine())

    assert isinstance(runtime.engine, UnavailableInferenceEngine)
    status = runtime.status()
    assert {
        key: status[key]
        for key in ("selected", "available", "loaded", "reason")
    } == {
        "selected": "unavailable",
        "available": False,
        "loaded": False,
        "reason": "fake unavailable",
    }
    assert status["gpu_preprocessor"] == {
        "selected": "",
        "enabled": False,
        "available": False,
        "reason": "disabled",
    }


def test_runtime_probes_engine_once_and_preserves_first_unavailable_reason() -> None:
    class FakeUnavailableEngine:
        engine_id = "fake"

        def __init__(self) -> None:
            self.available_calls = 0
            self.status_calls = 0
            self.reason = ""

        def available(self) -> bool:
            self.available_calls += 1
            self.reason = "first unavailable reason"
            return False

        def last_reason(self) -> str:
            return self.reason

        def status(self) -> dict:
            self.status_calls += 1
            self.reason = "status changed reason"
            return {
                "selected": self.engine_id, "available": False,
                "loaded": False, "reason": self.reason,
            }

        def load(self, artifact_path, classes, input_shape) -> None:
            raise AssertionError("not used")

        def infer(self, frame) -> InferenceResult:
            raise AssertionError("not used")

    engine = FakeUnavailableEngine()
    runtime = InferenceRuntime(engine)

    assert engine.available_calls == 1
    assert engine.status_calls == 0
    assert runtime.status()["reason"] == "first unavailable reason"


def test_missing_tensorrt_does_not_break_import_or_runtime_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError("No module named tensorrt")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    status = InferenceRuntime(TensorRtInferenceEngine()).status()

    assert status["selected"] == "unavailable"
    assert status["available"] is False
    assert status["loaded"] is False
    assert "TensorRT unavailable" in status["reason"]


class FakeInferenceRuntime:
    def status(self) -> dict:
        return {"selected": "fake", "available": True}

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        return InferenceResult(
            available=True,
            detections=[InferenceDetection(0, 0.9, 10, 20, 30, 40)],
            classes=["target"],
        )


class CandidateFilteringInference:
    def status(self) -> dict:
        return {"selected": "fake", "available": True}

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        return InferenceResult(
            available=True,
            detections=[
                InferenceDetection(1, 0.92, 300, 250, 340, 330),
                InferenceDetection(0, 0.99, 10, 10, 30, 30),
            ],
            classes=["body", "head"],
        )


class EmptyInferenceRuntime:
    def status(self) -> dict:
        return {"selected": "empty", "available": True}

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        return InferenceResult(available=True, detections=[], classes=[])


class TwoFramesThenEmptyInference:
    def __init__(self) -> None:
        self.calls = 0

    def status(self) -> dict:
        return {"selected": "sequence", "available": True}

    def infer(self, frame: CapturedFrame) -> InferenceResult:
        del frame
        self.calls += 1
        if self.calls == 1:
            return InferenceResult(
                available=True,
                detections=[InferenceDetection(0, 0.9, x=296, y=220, w=40, h=80)],
                classes=["target"],
            )
        if self.calls == 2:
            return InferenceResult(
                available=True,
                detections=[InferenceDetection(0, 0.9, x=304, y=220, w=40, h=80)],
                classes=["target"],
            )
        return InferenceResult(available=True, detections=[], classes=["target"])


class FakeKmNetExecutor:
    executor_id = "kmnet"

    def __init__(self) -> None:
        self.history: list[ControlOutput] = []

    def available(self) -> bool:
        return True

    def execute(self, output: ControlOutput) -> ExecutionResult:
        self.history.append(output)
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=True,
            intent=output,
            message="sent by fake kmNet executor",
            metadata={
                "stage": "fake_kmnet",
                "api_name": "move",
                "driver_rc": 0,
                "driver_dx": int(output.dx),
                "driver_dy": int(output.dy),
            },
        )


class FailingKmNetExecutor(FakeKmNetExecutor):
    def execute(self, output: ControlOutput) -> ExecutionResult:
        self.history.append(output)
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=False,
            intent=output,
            message="driver failed",
            metadata={
                "stage": "fake_kmnet",
                "api_name": "move",
                "driver_rc": -1,
                "driver_dx": int(output.dx),
                "driver_dy": int(output.dy),
            },
        )


class FakeControlFrameRecorder:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record_control_frame(self, record: dict) -> None:
        self.records.append(dict(record))


def _runtime_service(tmp_path, inference, *, recorder=None):
    kmnet = FakeKmNetExecutor()
    service = RuntimeService(
        config=RuntimeConfig(),
        models=ModelRegistry(tmp_path / "db.sqlite", tmp_path / "models"),
        executors=ExecutorRegistry(
            [kmnet],
            default="kmnet",
            scheduler=CommandScheduler(min_interval_s=0.0),
        ),
        inference=inference,
        recorder=recorder,
    )
    return service, kmnet


def _runtime_service_with_executor(tmp_path, inference, executor, *, recorder=None):
    service = RuntimeService(
        config=RuntimeConfig(),
        models=ModelRegistry(tmp_path / "db.sqlite", tmp_path / "models"),
        executors=ExecutorRegistry(
            [executor],
            default="kmnet",
            scheduler=CommandScheduler(min_interval_s=0.0, device_error_cooldown_s=0.25),
        ),
        inference=inference,
        recorder=recorder,
    )
    return service


def test_runtime_service_logs_production_chain_and_calibration_profile_on_startup(
    tmp_path,
    caplog,
) -> None:
    caplog.set_level("INFO", logger="novasight.runtime.service")

    _runtime_service(tmp_path, EmptyInferenceRuntime())

    assert "production_control_chain event=startup" in caplog.text
    assert (
        "CompensatedTarget(Control px)->AngularErrorMapper->AngularPDController->CommandScheduler->kmNet"
        in caplog.text
    )
    assert "calibration_profile_id=default" in caplog.text
    assert "calibration_profile_version=1" in caplog.text


def test_control_frame_csv_recorder_writes_fixed_contract_fields(tmp_path) -> None:
    path = tmp_path / "control_frames.csv"
    recorder = ControlFrameCsvRecorder(path)

    recorder.record_control_frame({"frame_id": 7, "error_x_rad": 0.25})
    recorder.close()

    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",") == CONTROL_FRAME_FIELDS
    assert "frame_id" in CONTROL_FRAME_FIELDS
    assert "candidate_count" in CONTROL_FRAME_FIELDS
    assert "error_x_rad" in CONTROL_FRAME_FIELDS
    assert "final_output_x_counts" in CONTROL_FRAME_FIELDS
    assert rows[1].startswith("7,")


def _control_frame_record(
    *,
    frame_id: int,
    capture_ts_ns: int,
    error_x_rad: float,
    error_y_rad: float = 0.0,
    command_status: str = "sent",
    track_id: int = 1,
) -> dict:
    record = {field: "" for field in CONTROL_FRAME_FIELDS}
    record.update(
        {
            "frame_id": frame_id,
            "capture_ts_ns": capture_ts_ns,
            "measurement_age_ms": 1.0,
            "inference_latency_ms": 2.0,
            "candidate_count": 1,
            "selected_class_id": 0,
            "selected_confidence": 0.9,
            "selected_quality_score": 0.8,
            "track_id": track_id,
            "track_state": "CONFIRMED",
            "identity_confidence": 0.95,
            "x": 320.0 + error_x_rad * 1000.0,
            "y": 320.0 + error_y_rad * 1000.0,
            "predicted": False,
            "raw_aim_x": 320.0,
            "raw_aim_y": 320.0,
            "smoothed_aim_x": 320.0,
            "smoothed_aim_y": 320.0,
            "control_width_px": 640,
            "control_height_px": 640,
            "error_x_px": error_x_rad * 1000.0,
            "error_y_px": error_y_rad * 1000.0,
            "error_x_rad": error_x_rad,
            "error_y_rad": error_y_rad,
            "u_x_rad": error_x_rad * 0.3,
            "u_y_rad": error_y_rad * 0.3,
            "raw_counts_x": error_x_rad * 100.0,
            "raw_counts_y": error_y_rad * 100.0,
            "residual_x_after": 0.25,
            "residual_y_after": 0.0,
            "final_output_x_counts": round(error_x_rad * 100.0),
            "final_output_y_counts": round(error_y_rad * 100.0),
            "sent_x": round(error_x_rad * 100.0),
            "sent_y": round(error_y_rad * 100.0),
            "command_status": command_status,
            "global_state": "sent" if command_status == "sent" else "not_emitted",
            "config_version": 1,
        }
    )
    return record


def test_control_frame_replay_uses_original_timestamps_and_injections() -> None:
    records = [
        _control_frame_record(frame_id=1, capture_ts_ns=1_000_000_000, error_x_rad=0.02),
        _control_frame_record(frame_id=2, capture_ts_ns=1_010_000_000, error_x_rad=0.01),
        _control_frame_record(frame_id=3, capture_ts_ns=1_020_000_000, error_x_rad=-0.01),
        _control_frame_record(frame_id=4, capture_ts_ns=1_030_000_000, error_x_rad=-0.02),
    ]

    result = ControlFrameReplay().run(
        records,
        injection=ReplayInjection(
            fixed_latency_ms=5.0,
            miss_frame_ids=frozenset({2}),
            lost_ranges=((3, 3),),
            track_switches={4: 99},
            device_fail_frame_ids=frozenset({4}),
            stale_after_ms=4.0,
        ),
    )

    assert [event.replay_offset_ns for event in result.events] == [
        0,
        10_000_000,
        20_000_000,
        30_000_000,
    ]
    assert [event.delta_ns for event in result.events] == [
        0,
        10_000_000,
        10_000_000,
        10_000_000,
    ]
    assert result.events[1].record["candidate_count"] == 0
    assert result.events[1].record["track_state"] == "PREDICTING"
    assert result.events[2].record["track_state"] == "LOST"
    assert result.events[2].record["cancel_reason"] == "TARGET_UNAVAILABLE"
    assert result.events[3].record["track_id"] == 99
    assert result.events[3].record["command_status"] == "device_error"
    assert result.events[3].record["global_state"] == "cooldown"
    assert result.metrics.frame_count == 4
    assert result.metrics.duration_ms == pytest.approx(30.0)
    assert result.metrics.predicted_count == 1
    assert result.metrics.missing_count >= 2
    assert result.metrics.switch_count == 1
    assert result.metrics.device_failure_count == 1
    assert result.metrics.stale_count == 4
    assert result.metrics.overshoot_count == 1


def test_control_frame_replay_loads_csv_and_compares_metrics(tmp_path) -> None:
    path = tmp_path / "control_frames.csv"
    recorder = ControlFrameCsvRecorder(path)
    recorder.record_control_frame(
        _control_frame_record(frame_id=1, capture_ts_ns=2_000_000_000, error_x_rad=0.03)
    )
    recorder.record_control_frame(
        _control_frame_record(frame_id=2, capture_ts_ns=2_020_000_000, error_x_rad=0.01)
    )
    recorder.close()
    replay = ControlFrameReplay()

    baseline = replay.run(replay.load_csv(path))
    variant = replay.run(
        [
            _control_frame_record(frame_id=1, capture_ts_ns=2_000_000_000, error_x_rad=0.02),
            _control_frame_record(frame_id=2, capture_ts_ns=2_020_000_000, error_x_rad=0.005),
        ]
    )
    comparison = compare_replay_metrics(baseline.metrics, variant.metrics)

    assert baseline.violations == ()
    assert baseline.events[1].replay_offset_ns == 20_000_000
    assert comparison.delta["p95_abs_error_x_rad"] < 0


def test_control_frame_replay_reports_timestamp_decrease() -> None:
    result = ControlFrameReplay().run(
        [
            _control_frame_record(frame_id=1, capture_ts_ns=2_000_000_000, error_x_rad=0.01),
            _control_frame_record(frame_id=2, capture_ts_ns=1_990_000_000, error_x_rad=0.01),
        ]
    )

    assert result.violations == ("timestamp_decreased:index=1",)
    assert result.metrics.timestamp_error_count == 1


def test_default_replay_acceptance_suite_covers_design_scenarios() -> None:
    report = run_replay_acceptance()
    case_names = {case.name for case in build_default_replay_acceptance_cases()}

    assert report.passed is True
    assert {
        "static_target",
        "constant_velocity",
        "sudden_turn",
        "single_frame_miss",
        "long_lost",
        "track_switch",
        "device_failure",
        "aspect_ratio_switch",
        "calibration_change",
    } <= case_names
    summary = report.as_dict()
    assert summary["passed"] is True
    assert summary["cases"]["single_frame_miss"]["metrics"]["predicted_count"] >= 1
    assert summary["cases"]["long_lost"]["metrics"]["missing_count"] >= 3
    assert summary["cases"]["track_switch"]["metrics"]["switch_count"] >= 1
    assert summary["cases"]["device_failure"]["metrics"]["device_failure_count"] >= 1


def test_replay_acceptance_gate_reports_threshold_failures() -> None:
    bad_case = ReplayAcceptanceCase(
        name="bad_static",
        records=(
            _control_frame_record(frame_id=1, capture_ts_ns=1_000_000_000, error_x_rad=0.10),
            _control_frame_record(frame_id=2, capture_ts_ns=1_010_000_000, error_x_rad=0.08),
        ),
        gate=ReplayAcceptanceGate(max_p95_abs_error_x_rad=0.01),
    )

    report = run_replay_acceptance((bad_case,))

    assert report.passed is False
    assert report.reports[0].case_name == "bad_static"
    assert any("p95_abs_error_x_rad" in failure for failure in report.reports[0].failures)


def test_candidate_filter_records_filter_reasons_and_quality() -> None:
    context = FrameContext(
        frame_id=9,
        capture_ts_ns=123_456,
        width=640,
        height=640,
        detections=[
            Detection(1, 0.9, x=300, y=260, w=40, h=80),
            Detection(2, 0.9, x=300, y=260, w=40, h=80),
            Detection(1, 0.9, x=300, y=260, w=400, h=20),
            Detection(1, 0.2, x=300, y=260, w=40, h=80),
            Detection(1, 0.9, x=10, y=10, w=20, h=20),
        ],
    )
    result = CandidateFilter(
        CandidateFilterConfig(
            allowed_class_ids={1},
            min_confidence=0.5,
            selection_fov=SelectionFovConfig(radius_x_percent=35, radius_y_percent=35),
            ratio_check=RatioCheckConfig(max_aspect_ratio=6.0),
        ),
        quality_scorer=QualityScorer(QualityScoreConfig(confidence_weight=1.0, area_weight=1.0)),
    ).apply(context, aim_ratio=50)

    assert len(result.candidates) == 1
    accepted = result.candidates[0]
    assert accepted.frame_id == 9
    assert accepted.capture_ts_ns == 123_456
    assert accepted.selection_fov_pass is True
    assert accepted.ratio_valid is True
    assert accepted.quality.conf_score == pytest.approx(0.9)
    assert 0 < accepted.quality.area_score < 1
    assert {item.reason for item in result.rejected} == {
        "class_filter",
        "confidence_filter",
        "selection_fov",
        "ratio_check",
    }


def _scored_candidates(context: FrameContext):
    return CandidateFilter(
        CandidateFilterConfig(
            selection_fov=SelectionFovConfig(radius_x_percent=100, radius_y_percent=100),
        )
    ).apply(context, aim_ratio=50).candidates


def test_runtime_tracker_requires_new_frame_before_confirming() -> None:
    tracker = RuntimeTracker(TrackerConfig(confirm_frames=2))
    first = FrameContext(
        frame_id=1,
        capture_ts_ns=100,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=300, y=300, w=40, h=40)],
    )

    first_update = tracker.update(first, _scored_candidates(first))
    repeat_update = tracker.update(first, _scored_candidates(first))
    second = FrameContext(
        frame_id=2,
        capture_ts_ns=200,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=302, y=300, w=40, h=40)],
    )
    second_update = tracker.update(second, _scored_candidates(second))

    assert first_update.tracks == []
    assert first_update.state == "ACQUIRING"
    assert repeat_update.tracks == []
    assert repeat_update.debug["repeat_frame"] is True
    assert len(second_update.tracks) == 1
    assert second_update.tracks[0].track_id == 1


def test_runtime_tracker_identity_uncertain_pauses_confirmed_track() -> None:
    tracker = RuntimeTracker(
        TrackerConfig(
            confirm_frames=2,
            matching_distance_px=100,
            ambiguity_margin=0.2,
            match_threshold=1.0,
        )
    )
    first = FrameContext(
        frame_id=1,
        capture_ts_ns=100,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=300, y=300, w=40, h=40)],
    )
    second = FrameContext(
        frame_id=2,
        capture_ts_ns=200,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=302, y=300, w=40, h=40)],
    )
    ambiguous = FrameContext(
        frame_id=3,
        capture_ts_ns=300,
        width=640,
        height=640,
        detections=[
            Detection(0, 0.9, x=292, y=300, w=40, h=40),
            Detection(0, 0.9, x=312, y=300, w=40, h=40),
        ],
    )

    tracker.update(first, _scored_candidates(first))
    confirmed = tracker.update(second, _scored_candidates(second))
    uncertain = tracker.update(ambiguous, _scored_candidates(ambiguous))

    assert len(confirmed.tracks) == 1
    assert uncertain.tracks == []
    assert uncertain.state == "IDENTITY_UNCERTAIN"
    assert uncertain.debug["identity_uncertain_tracks"] == [1]


def test_kalman_estimator_rejects_extreme_nis() -> None:
    estimator = KalmanEstimator(
        track_id=1,
        x=100,
        y=100,
        ts_ns=1_000_000_000,
        config=KalmanConfig(
            measurement_noise_x=1,
            measurement_noise_y=1,
            nis_threshold=9.21,
            nis_hard_reject=16,
        ),
    )

    result = estimator.update(
        measurement_x=500,
        measurement_y=500,
        ts_ns=1_010_000_000,
        identity_confidence=1.0,
    )

    assert result.valid is False
    assert result.reason == "NIS_REJECT"
    assert result.nis > 16


def test_runtime_tracker_uses_kalman_nis_gate_for_association() -> None:
    tracker = RuntimeTracker(
        TrackerConfig(
            confirm_frames=2,
            matching_distance_px=1000,
            mahalanobis_gate=2.0,
            kalman=KalmanConfig(
                measurement_noise_x=4,
                measurement_noise_y=4,
                nis_threshold=9.21,
                nis_hard_reject=16,
            ),
        )
    )
    first = FrameContext(
        frame_id=1,
        capture_ts_ns=1_000_000_000,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=80, y=80, w=40, h=40)],
    )
    second = FrameContext(
        frame_id=2,
        capture_ts_ns=1_010_000_000,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=82, y=80, w=40, h=40)],
    )
    jump = FrameContext(
        frame_id=3,
        capture_ts_ns=1_020_000_000,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=500, y=500, w=40, h=40)],
    )

    tracker.update(first, _scored_candidates(first))
    confirmed = tracker.update(second, _scored_candidates(second))
    gated = tracker.update(jump, _scored_candidates(jump))

    assert len(confirmed.tracks) == 1
    assert gated.tracks == []
    assert gated.state == "TARGET_UNAVAILABLE"
    states = {item["track_id"]: item["state"] for item in gated.debug["tracks"]}
    assert states[1] == "MISSING"
    assert any(item["state"] == "TENTATIVE" for item in gated.debug["tracks"])
    assert gated.debug["config"]["mahalanobis_gate"] == 2.0
    assert gated.debug["tracks"][0]["estimate"]["predicted"] is True


def test_runtime_tracker_exposes_design_stop_states_for_unavailable_reasons() -> None:
    tracker = RuntimeTracker(TrackerConfig(confirm_frames=1))
    context = FrameContext(
        frame_id=1,
        capture_ts_ns=1_000_000_000,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=300, y=300, w=40, h=40)],
    )
    tracker.update(context, _scored_candidates(context))

    cooldown = tracker.mark_unavailable(
        "COOLDOWN",
        context=FrameContext(frame_id=2, capture_ts_ns=1_010_000_000, width=640, height=640),
        reason="device error cooldown",
    )
    assert cooldown.tracks == []
    assert cooldown.state == "COOLDOWN"
    assert cooldown.debug["state"] == "COOLDOWN"
    assert cooldown.debug["tracks"][0]["state"] == "LOST"

    disabled = tracker.mark_unavailable(
        "DISABLED",
        context=FrameContext(frame_id=3, capture_ts_ns=1_020_000_000, width=640, height=640),
        reason="runtime disabled",
    )
    assert disabled.tracks == []
    assert disabled.state == "DISABLED"
    assert disabled.debug["state"] == "DISABLED"


def test_runtime_tracker_exposes_valid_missing_track_as_predicting() -> None:
    tracker = RuntimeTracker(
        TrackerConfig(
            confirm_frames=2,
            matching_distance_px=100,
            missing_timeout_ms=120,
            kalman=KalmanConfig(min_prediction_confidence=0.01),
        )
    )
    first = FrameContext(
        frame_id=1,
        capture_ts_ns=1_000_000_000,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=296, y=220, w=40, h=80)],
    )
    second = FrameContext(
        frame_id=2,
        capture_ts_ns=1_010_000_000,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=304, y=220, w=40, h=80)],
    )
    missing = FrameContext(
        frame_id=3,
        capture_ts_ns=1_020_000_000,
        width=640,
        height=640,
        detections=[],
    )

    tracker.update(first, _scored_candidates(first))
    confirmed = tracker.update(second, _scored_candidates(second))
    predicted = tracker.update(missing, _scored_candidates(missing))

    assert confirmed.state == "TRACKING"
    assert predicted.state == "PREDICTING"
    assert len(predicted.tracks) == 1
    assert predicted.tracks[0].track_id == 1
    assert predicted.tracks[0].cx > confirmed.tracks[0].cx
    track_debug = predicted.debug["tracks"][0]
    assert track_debug["state"] == "PREDICTING"
    assert track_debug["estimate"]["predicted"] is True
    assert track_debug["estimate"]["valid"] is True


def test_runtime_target_selector_requires_committed_switch_before_new_target_controls() -> None:
    selector = RuntimeTargetSelector()

    first = FrameContext(
        frame_id=1,
        capture_ts_ns=1_000_000_000,
        width=640,
        height=640,
        detections=[Detection(0, 0.9, x=300, y=260, w=40, h=80)],
    )
    pending = FrameContext(
        frame_id=2,
        capture_ts_ns=1_010_000_000,
        width=640,
        height=640,
        detections=[
            Detection(0, 0.9, x=300, y=260, w=40, h=80),
            Detection(1, 0.9, x=302, y=260, w=40, h=80),
        ],
    )
    committed = FrameContext(
        frame_id=3,
        capture_ts_ns=1_020_000_000,
        width=640,
        height=640,
        detections=[
            Detection(0, 0.9, x=300, y=260, w=40, h=80),
            Detection(1, 0.9, x=304, y=260, w=40, h=80),
        ],
    )

    first_selection = selector.select(
        first,
        min_confidence=0.0,
        fov_ratio=1.0,
        class_priority=[1, 0],
        class_priority_quality_margin=0.2,
        tracker_confirm_frames=1,
        target_switch_min_preference_advantage=0.05,
        target_switch_min_continuity_score=0.0,
        target_switch_confirm_frames=2,
    )
    pending_selection = selector.select(
        pending,
        min_confidence=0.0,
        fov_ratio=1.0,
        class_priority=[1, 0],
        class_priority_quality_margin=0.2,
        tracker_confirm_frames=1,
        target_switch_min_preference_advantage=0.05,
        target_switch_min_continuity_score=0.0,
        target_switch_confirm_frames=2,
    )
    pending_debug = dict(selector.last_debug["switch"])
    committed_selection = selector.select(
        committed,
        min_confidence=0.0,
        fov_ratio=1.0,
        class_priority=[1, 0],
        class_priority_quality_margin=0.2,
        tracker_confirm_frames=1,
        target_switch_min_preference_advantage=0.05,
        target_switch_min_continuity_score=0.0,
        target_switch_confirm_frames=2,
    )

    assert first_selection.target is not None
    assert first_selection.target.cls == 0
    assert pending_selection.target is None
    assert pending_selection.state == "switch_pending"
    assert pending_debug["state"] == "pending"
    assert pending_debug["frames"] == 1
    assert committed_selection.target is not None
    assert committed_selection.target.cls == 1
    assert committed_selection.state == "switch_committed"
    assert "SWITCH_COMMITTED" in committed_selection.reason


def test_runtime_service_selector_uses_candidate_quality_debug(tmp_path) -> None:
    service, _kmnet = _runtime_service(tmp_path, CandidateFilteringInference())
    service.config.control.fov_ratio = 0.4
    service.config.control.min_confidence = 0.5
    service.config.control.candidate_ratio_max_aspect = 6.0
    service.config.control.class_priority_quality_margin = 1.0
    service.config.inference.detection_class_priority = "1,0"
    service.update_config(service.config)
    ts_ns = time.monotonic_ns()
    first = CapturedFrame(7, 640, 640, "BGR", ts_ns, 1.0, image=None)
    second = CapturedFrame(8, 640, 640, "BGR", ts_ns + 10_000_000, 1.0, image=None)

    service.process_captured_frame(first)
    assert service.last_target is None
    assert service.last_control["selector_state"] == "acquiring"

    service.process_control_tick()
    assert service.last_target is None
    assert service.last_control["selector_state"] == "acquiring"

    service.process_captured_frame(second)

    assert service.last_target is not None
    assert service.last_target["class_id"] == 1
    selector_debug = service.last_control["selector_debug"]
    assert selector_debug["raw_candidates"] == 2
    assert selector_debug["filtered_candidates"] == 1
    assert selector_debug["inside_fov"] == 1
    assert selector_debug["rejected"][0]["reason"] == "selection_fov"
    assert selector_debug["selected"]["quality_score"] > 0
    candidate_filter = service.last_control["candidate_filter"]
    assert candidate_filter["raw_candidates"] == 2
    assert candidate_filter["filtered_candidates"] == 1
    assert candidate_filter["inside_fov"] == 1
    assert candidate_filter["rejected_candidates"] == 1
    assert candidate_filter["rejected"][0]["reason"] == "selection_fov"
    assert candidate_filter["selected"]["quality_score"] == pytest.approx(
        selector_debug["selected"]["quality_score"]
    )
    assert service.last_target["candidate_filter"]["selected"]["quality_score"] == pytest.approx(
        selector_debug["selected"]["quality_score"]
    )
    assert service.last_control["quality_score"] == pytest.approx(selector_debug["selected"]["quality_score"])
    track_diagnostics = service.last_control["track_diagnostics"]
    assert track_diagnostics["selected_track_id"] == service.last_target["track_id"]
    assert track_diagnostics["selected_track_state"] == "CONFIRMED"
    assert track_diagnostics["selected_continuity_score"] is not None
    assert track_diagnostics["selected_identity_confidence"] is not None
    assert track_diagnostics["selected_missing_ms"] == pytest.approx(0.0)
    assert track_diagnostics["switch_reason"]
    assert service.last_target["track_diagnostics"]["selected_track_id"] == track_diagnostics["selected_track_id"]

def test_runtime_service_publishes_inference_to_context_and_target(tmp_path) -> None:
    service, kmnet = _runtime_service(tmp_path, FakeInferenceRuntime())
    service.config.control.fov_ratio = 1.0
    ts_ns = time.monotonic_ns()
    first = CapturedFrame(7, 640, 480, "BGR", ts_ns, 1.0, image=None)
    second = CapturedFrame(8, 640, 480, "BGR", ts_ns + 10_000_000, 1.0, image=None)

    service.process_captured_frame(first)
    assert service.last_target is None
    assert service.last_control["selector_state"] == "acquiring"

    service.process_captured_frame(second)

    # The service must turn the inference result into a populated
    # FrameContext and a target selection; downstream strategy may
    # decide whether to emit an intent based on the hardware trigger.
    assert service.last_frame_context is not None
    assert service.last_frame_context.frame_id == 8
    assert service.last_frame_context.detections
    assert service.last_target is not None
    assert service.last_inference_status["available"] is True
    assert kmnet.history == []  # no trigger configured


def test_runtime_service_clears_pending_scheduler_command_when_target_unavailable(tmp_path) -> None:
    service, _kmnet = _runtime_service(tmp_path, EmptyInferenceRuntime())
    scheduler = service.executors.scheduler
    assert scheduler is not None
    scheduler.min_interval_s = 0.1
    scheduler.submit(
        ControlOutput(
            dx=8,
            dy=0,
            action="move",
            confidence=1.0,
            source_id="test",
            accepted=True,
            clipped=False,
            reason="seed pending",
            source_frame_id=1,
            source_track_id=1,
        ),
        now_s=1.0,
    )
    scheduler.submit(
        ControlOutput(
            dx=9,
            dy=0,
            action="move",
            confidence=1.0,
            source_id="test",
            accepted=True,
            clipped=False,
            reason="held pending",
            source_frame_id=1,
            source_track_id=1,
        ),
        now_s=1.001,
    )
    assert scheduler.status(now_s=1.001)["has_pending"] is True

    service.process_captured_frame(
        CapturedFrame(9, 640, 480, "BGR", time.monotonic_ns(), 1.0, image=None)
    )

    status = scheduler.status(now_s=1.002)
    assert status["has_pending"] is False
    assert status["last_cancel_reason"] == "TARGET_UNAVAILABLE"


def test_runtime_config_non_calibration_update_preserves_selector_state(tmp_path) -> None:
    service, _kmnet = _runtime_service(tmp_path, EmptyInferenceRuntime())
    reset_calls: list[str] = []
    original_reset = service.target_selector.reset

    def tracked_reset() -> None:
        reset_calls.append("reset")
        original_reset()

    service.target_selector.reset = tracked_reset  # type: ignore[method-assign]
    updated = copy.deepcopy(service.config)
    updated.inference.confidence_threshold = 0.33

    service.update_config(updated)

    assert reset_calls == []
    assert service.last_control is None


def test_runtime_config_calibration_change_resets_control_state_and_scheduler(tmp_path) -> None:
    service, _kmnet = _runtime_service(tmp_path, EmptyInferenceRuntime())
    service.last_frame_context = FrameContext(
        frame_id=21,
        width=640,
        height=640,
        capture_ts_ns=time.monotonic_ns(),
    )
    service.last_target = {"target_key": "track:1"}
    service.last_execution = {"sent": True}
    scheduler = service.executors.scheduler
    assert scheduler is not None
    scheduler.min_interval_s = 0.1
    scheduler.submit(
        ControlOutput(
            dx=8,
            dy=0,
            action="move",
            confidence=1.0,
            source_id="test",
            accepted=True,
            clipped=False,
            reason="seed pending",
            source_frame_id=1,
            source_track_id=1,
        ),
        now_s=1.0,
    )
    scheduler.submit(
        ControlOutput(
            dx=9,
            dy=0,
            action="move",
            confidence=1.0,
            source_id="test",
            accepted=True,
            clipped=False,
            reason="held pending",
            source_frame_id=1,
            source_track_id=1,
        ),
        now_s=1.001,
    )
    assert scheduler.status(now_s=1.001)["has_pending"] is True

    updated = copy.deepcopy(service.config)
    updated.calibration.fov_x_deg = 103.0
    updated.calibration.counts_per_360_x = 9900.0
    updated.calibration.axis_sign_y = -1.0
    updated.calibration.game_sensitivity_fingerprint = "arena-103-v2"
    updated.roi.size = 320

    service.update_config(updated)

    assert scheduler.status(now_s=1.002)["has_pending"] is False
    assert scheduler.status(now_s=1.002)["last_cancel_reason"] == "CALIBRATION_PROFILE_CHANGED"
    assert service.executors.status()["scheduler"]["last_cancel_reason"] == "CALIBRATION_PROFILE_CHANGED"
    assert service.last_frame_context is None
    assert service.last_target is None
    assert service.last_execution is None
    assert service.last_control == {
        "will_emit": False,
        "control_allowed": False,
        "global_state": "DISABLED",
        "runtime_reset": True,
        "runtime_reset_reason": "CALIBRATION_PROFILE_CHANGED",
        "selector_state": "reset",
        "selection_reason": "CALIBRATION_PROFILE_CHANGED",
        "calibration_status": {
            "state": "unverified",
            "control_allowed": True,
            "expected_fingerprint": "arena-103-v2",
            "observed_fingerprint": "",
            "source": "",
            "updated_ts_ns": 0,
            "reason_code": "",
            "reason": "external sensitivity fingerprint has not been reported",
        },
        "candidate_filter": {
            "raw_candidates": 0,
            "filtered_candidates": 0,
            "inside_fov": 0,
            "rejected_candidates": 0,
            "candidates": [],
            "rejected": [],
            "selected": None,
            "reason": "",
        },
    }


def test_runtime_service_blocks_control_when_external_sensitivity_fingerprint_mismatches(tmp_path) -> None:
    service, kmnet = _runtime_service(tmp_path, FakeInferenceRuntime())
    service.config.control.fov_ratio = 1.0
    service.config.control.trigger_mode = "always"
    service.config.calibration.game_sensitivity_fingerprint = "arena-105-live"
    service.update_config(service.config)

    status = service.update_external_sensitivity_fingerprint(
        "arena-105-stale",
        source="game",
    )
    assert status["state"] == "mismatch"
    assert status["control_allowed"] is False

    ts_ns = time.monotonic_ns()
    service.process_captured_frame(CapturedFrame(7, 640, 480, "BGR", ts_ns, 1.0, image=None))
    service.process_captured_frame(CapturedFrame(8, 640, 480, "BGR", ts_ns + 10_000_000, 1.0, image=None))
    result = service.process_control_tick()

    assert result.control_intents == []
    assert result.execution_results == []
    assert kmnet.history == []
    assert service.last_target is not None
    assert service.last_control["control_allowed"] is False
    assert service.last_control["will_emit"] is False
    assert service.last_control["reason"] == "CALIBRATION_FINGERPRINT_MISMATCH"
    assert service.last_control["calibration_status"]["state"] == "mismatch"
    assert service.last_execution["sent"] is False
    assert service.last_execution["message"] == "external sensitivity fingerprint does not match calibration profile"
    assert service.state().vision["calibration"]["state"] == "mismatch"
    control_stage = next(
        stage for stage in service.state().vision["trace"]["stages"] if stage["id"] == "control"
    )
    assert control_stage["status"] == "blocked"
    assert control_stage["message"] == "external sensitivity fingerprint does not match calibration profile"


def test_runtime_service_emits_clamped_intent_with_always_trigger(tmp_path) -> None:
    recorder = FakeControlFrameRecorder()
    service, kmnet = _runtime_service(tmp_path, FakeInferenceRuntime(), recorder=recorder)
    service.config.control.fov_ratio = 1.0
    service.config.control.trigger_mode = "always"
    service.config.control.experimental_angle_kalman_enabled = False
    service.config.consumers.recording = True
    service.update_config(service.config)
    ts_ns = time.monotonic_ns()
    first = CapturedFrame(7, 640, 480, "BGR", ts_ns, 1.0, image=None)
    second = CapturedFrame(8, 640, 480, "BGR", ts_ns + 10_000_000, 1.0, image=None)
    first_observation = service.process_captured_frame(first)
    observation = service.process_captured_frame(second)
    result = service.process_control_tick()

    assert first_observation.control_intents == []
    assert observation.control_intents == []
    assert len(result.control_intents) == 1
    assert len(result.execution_results) == 1
    output = result.execution_results[0].intent
    # The experimental angular strategy clamps before the policy layer.
    assert output.accepted is True
    assert output.clipped is False
    assert abs(output.dx) <= service.config.control.experimental_angle_max_step_counts
    assert abs(output.dy) <= service.config.control.experimental_angle_max_step_counts
    assert kmnet.history == [output]
    assert service.last_control["execution"]["sent"] is True
    assert service.last_control["execution_executor"] == "kmnet"
    assert service.last_control["driver_api"] == "move"
    assert service.last_control["driver_rc"] == 0
    assert service.last_control["driver_counts"] == {
        "dx": float(output.dx),
        "dy": float(output.dy),
    }
    assert recorder.records
    last_record = recorder.records[-1]
    assert last_record["frame_id"] == 8
    assert last_record["capture_ts_ns"] == second.capture_ts_ns
    assert last_record["candidate_count"] >= 1
    assert last_record["selected_class_id"] == 0
    assert last_record["track_id"] != ""
    assert last_record["error_x_rad"] != ""
    assert last_record["raw_counts_x"] != ""
    assert last_record["final_output_x_counts"] != ""
    assert last_record["sent_x"] == output.dx
    assert last_record["driver_x_counts"] == output.dx
    assert last_record["command_status"] == "sent"


def test_runtime_service_reports_cooldown_global_state_after_device_error(tmp_path) -> None:
    recorder = FakeControlFrameRecorder()
    failing = FailingKmNetExecutor()
    service = _runtime_service_with_executor(
        tmp_path,
        FakeInferenceRuntime(),
        failing,
        recorder=recorder,
    )
    service.config.control.fov_ratio = 1.0
    service.config.control.trigger_mode = "always"
    service.config.consumers.recording = True
    service.update_config(service.config)
    ts_ns = time.monotonic_ns()

    service.process_captured_frame(
        CapturedFrame(7, 640, 480, "BGR", ts_ns, 1.0, image=None)
    )
    service.process_captured_frame(
        CapturedFrame(8, 640, 480, "BGR", ts_ns + 10_000_000, 1.0, image=None)
    )
    result = service.process_control_tick()

    assert result.control_intents
    assert result.execution_results
    assert result.execution_results[0].sent is False
    assert service.last_control["global_state"] == "COOLDOWN"
    assert service.last_control["execution_sent"] is False
    assert service.last_control["execution"]["metadata"]["scheduler"]["execution"]["command_status"] == "error"
    assert service.executors.status()["scheduler"]["cooldown_active"] is True
    assert recorder.records[-1]["global_state"] == "COOLDOWN"
    assert recorder.records[-1]["command_status"] == "error"


def test_runtime_service_uses_predicted_track_for_limited_control_after_single_miss(tmp_path) -> None:
    service, kmnet = _runtime_service(tmp_path, TwoFramesThenEmptyInference())
    service.config.control.fov_ratio = 1.0
    service.config.control.trigger_mode = "always"
    service.config.control.kalman_min_prediction_confidence = 0.01
    service.config.control.scheduler_predicted_ttl_ms = 18
    service.update_config(service.config)
    ts_ns = time.monotonic_ns()

    service.process_captured_frame(CapturedFrame(7, 640, 480, "BGR", ts_ns, 1.0, image=None))
    service.process_captured_frame(CapturedFrame(8, 640, 480, "BGR", ts_ns + 10_000_000, 1.0, image=None))
    service.process_captured_frame(CapturedFrame(9, 640, 480, "BGR", ts_ns + 20_000_000, 1.0, image=None))
    result = service.process_control_tick()

    assert service.last_control["selector_state"] == "predicting"
    assert service.last_control["pipeline"]["tracker"]["new_observation"] is False
    assert service.last_control["pipeline"]["compensated_target"]["predicted_source"] is True
    track_diagnostics = service.last_control["track_diagnostics"]
    assert track_diagnostics["tracker_state"] == "PREDICTING"
    assert track_diagnostics["selected_track_state"] == "PREDICTING"
    assert track_diagnostics["selected_missing_ms"] is not None
    assert track_diagnostics["selected_continuity_score"] is not None
    assert service.last_target["track_diagnostics"]["selected_track_id"] == track_diagnostics["selected_track_id"]
    assert len(result.control_intents) == 1
    assert result.control_intents[0].predicted_source is True
    assert kmnet.history[-1].predicted_source is True
    scheduler_status = service.executors.status()["scheduler"]
    assert scheduler_status["predicted_ttl_ms"] == pytest.approx(18.0)


def test_runtime_service_stops_prediction_and_clears_pending_after_prediction_threshold(tmp_path) -> None:
    service, kmnet = _runtime_service(tmp_path, TwoFramesThenEmptyInference())
    service.config.control.fov_ratio = 1.0
    service.config.control.trigger_mode = "always"
    service.config.control.kalman_min_prediction_confidence = 0.01
    service.config.control.kalman_max_predict_missing_ms = 25
    service.update_config(service.config)
    scheduler = service.executors.scheduler
    assert scheduler is not None
    ts_ns = time.monotonic_ns()

    service.process_captured_frame(CapturedFrame(7, 640, 480, "BGR", ts_ns, 1.0, image=None))
    service.process_captured_frame(CapturedFrame(8, 640, 480, "BGR", ts_ns + 10_000_000, 1.0, image=None))
    service.process_captured_frame(CapturedFrame(9, 640, 480, "BGR", ts_ns + 20_000_000, 1.0, image=None))
    predicted = service.process_control_tick()
    assert service.last_control["selector_state"] == "predicting"
    assert predicted.control_intents
    assert kmnet.history[-1].predicted_source is True

    scheduler.min_interval_s = 0.1
    scheduler.submit(
        ControlOutput(
            dx=8,
            dy=0,
            action="move",
            confidence=1.0,
            source_id="test",
            accepted=True,
            clipped=False,
            reason="seed prediction pending",
            source_frame_id=9,
            source_track_id=1,
            predicted_source=True,
        ),
        now_s=1.0,
    )
    scheduler.submit(
        ControlOutput(
            dx=9,
            dy=0,
            action="move",
            confidence=1.0,
            source_id="test",
            accepted=True,
            clipped=False,
            reason="held prediction pending",
            source_frame_id=9,
            source_track_id=1,
            predicted_source=True,
        ),
        now_s=1.001,
    )
    assert scheduler.status(now_s=1.001)["has_pending"] is True

    expired = service.process_captured_frame(CapturedFrame(10, 640, 480, "BGR", ts_ns + 100_000_000, 1.0, image=None))

    assert service.last_control["selector_state"] != "predicting"
    tracker_debug = service.last_control["selector_debug"]["tracker"]
    assert service.last_control["selector_state"] == "target_unavailable"
    assert service.last_control["global_state"] == "TARGET_UNAVAILABLE"
    assert tracker_debug["state"] == "TARGET_UNAVAILABLE"
    assert tracker_debug["tracks"][0]["estimate"]["valid"] is False
    assert tracker_debug["tracks"][0]["state"] == "MISSING"
    assert expired.control_intents == []
    scheduler_status = scheduler.status(now_s=1.002)
    assert scheduler_status["has_pending"] is False
    assert scheduler_status["last_cancel_reason"] == "TARGET_UNAVAILABLE"


def test_aim_point_generator_applies_percent_offsets_and_ema() -> None:
    generator = AimPointGenerator()
    cfg = AimPointConfig(
        horizontal_percent=25,
        vertical_percent_from_top=40,
        offset_x_px=3,
        offset_y_px=-2,
        ema_enabled=True,
        ema_alpha=0.5,
    )
    first_track = Track(track_id=1, cls=0, score=0.9, x1=100, y1=50, x2=180, y2=150)
    first_estimate = EstimatedTargetState(
        track_id=1,
        state_ts_ns=1_000,
        capture_ts_ns=900,
        x=140,
        y=100,
        vx=0,
        vy=0,
        valid=True,
    )
    first = generator.update(track=first_track, estimate=first_estimate, config=cfg)

    assert first.raw_x == pytest.approx(123)
    assert first.raw_y == pytest.approx(88)
    assert first.smoothed_x == pytest.approx(123)
    assert first.smoothed_y == pytest.approx(88)

    second_track = Track(track_id=1, cls=0, score=0.9, x1=120, y1=50, x2=200, y2=150)
    second_estimate = EstimatedTargetState(
        track_id=1,
        state_ts_ns=2_000,
        capture_ts_ns=1_900,
        x=160,
        y=100,
        vx=0,
        vy=0,
        valid=True,
    )
    second = generator.update(track=second_track, estimate=second_estimate, config=cfg)

    assert second.raw_x == pytest.approx(143)
    assert second.raw_y == pytest.approx(88)
    assert second.smoothed_x == pytest.approx(133)
    assert second.smoothed_y == pytest.approx(88)


def test_aim_point_generator_freezes_on_anchor_jump() -> None:
    generator = AimPointGenerator()
    cfg = AimPointConfig(
        horizontal_percent=50,
        vertical_percent_from_top=10,
        ema_enabled=True,
        ema_alpha=1.0,
        max_anchor_jump_ratio=0.05,
    )
    first_track = Track(track_id=2, cls=0, score=0.9, x1=100, y1=100, x2=140, y2=200)
    first_estimate = EstimatedTargetState(
        track_id=2,
        state_ts_ns=1_000,
        capture_ts_ns=900,
        x=120,
        y=150,
        vx=0,
        vy=0,
        valid=True,
    )
    first = generator.update(track=first_track, estimate=first_estimate, config=cfg)
    second_track = Track(track_id=2, cls=0, score=0.9, x1=100, y1=100, x2=140, y2=300)
    second_estimate = EstimatedTargetState(
        track_id=2,
        state_ts_ns=2_000,
        capture_ts_ns=1_900,
        x=120,
        y=200,
        vx=0,
        vy=0,
        valid=True,
    )
    second = generator.update(track=second_track, estimate=second_estimate, config=cfg)

    assert second.ema_reset_reason == "ANCHOR_JUMP"
    assert second.aim_confidence < 1
    assert second.smoothed_x == pytest.approx(first.smoothed_x)
    assert second.smoothed_y == pytest.approx(first.smoothed_y)


def test_latency_compensator_uses_state_timestamp_and_vector_clamp() -> None:
    aim = AimPointGenerator().update(
        track=Track(track_id=3, cls=0, score=0.9, x1=300, y1=300, x2=340, y2=340),
        estimate=EstimatedTargetState(
            track_id=3,
            state_ts_ns=10_000_000,
            capture_ts_ns=8_000_000,
            x=320,
            y=320,
            vx=1000,
            vy=1000,
            valid=True,
            prediction_confidence=1.0,
            velocity_measurements=5,
        ),
        config=AimPointConfig(ema_enabled=False),
    )
    estimate = EstimatedTargetState(
        track_id=3,
        state_ts_ns=10_000_000,
        capture_ts_ns=8_000_000,
        x=320,
        y=320,
        vx=1000,
        vy=1000,
        valid=True,
        prediction_confidence=1.0,
        velocity_measurements=5,
    )

    result = LatencyCompensator().compensate(
        aim=aim,
        estimate=estimate,
        source_frame_id=7,
        compute_ts_ns=30_000_000,
        roi_offset_x=100,
        roi_offset_y=50,
        roi_width=640,
        roi_height=640,
        config=LatencyCompensationConfig(
            scale=1.0,
            max_compensation_ms=100,
            reject_if_age_exceeds_ms=100,
            max_compensation_px=10,
            min_velocity_px_s=0,
            max_velocity_px_s=3000,
            min_velocity_measurements=2,
            min_velocity_confidence=0,
            estimated_actuation_delay_ms=0,
        ),
    )

    assert result.applied is True
    assert result.control_allowed is True
    assert result.compensation_ms == pytest.approx(20)
    assert (result.delta_x**2 + result.delta_y**2) ** 0.5 == pytest.approx(10)
    assert result.control_x == pytest.approx(100 + result.roi_x)
    assert result.control_y == pytest.approx(50 + result.roi_y)


def test_latency_compensator_uses_coordinate_transform_for_control_point() -> None:
    aim = AimPointGenerator().update(
        track=Track(track_id=33, cls=0, score=0.9, x1=300, y1=300, x2=340, y2=340),
        estimate=EstimatedTargetState(
            track_id=33,
            state_ts_ns=10_000_000,
            capture_ts_ns=8_000_000,
            x=320,
            y=320,
            vx=0,
            vy=0,
            valid=True,
        ),
        config=AimPointConfig(ema_enabled=False),
    )
    transform = CoordinateTransform(
        model_width=640,
        model_height=640,
        roi_x=100,
        roi_y=50,
        roi_width=640,
        roi_height=640,
        capture_width=1920,
        capture_height=1080,
        control_origin_x=10,
        control_origin_y=20,
    )

    result = LatencyCompensator().compensate(
        aim=aim,
        estimate=EstimatedTargetState(
            track_id=33,
            state_ts_ns=10_000_000,
            capture_ts_ns=8_000_000,
            x=320,
            y=320,
            vx=0,
            vy=0,
            valid=True,
        ),
        source_frame_id=73,
        compute_ts_ns=12_000_000,
        roi_offset_x=100,
        roi_offset_y=50,
        roi_width=640,
        roi_height=640,
        control_width=1920,
        control_height=1080,
        config=LatencyCompensationConfig(enabled=False),
        coordinate_transform=transform,
    )

    expected = transform.roi_to_capture_point(aim.smoothed_x, aim.smoothed_y)
    expected = transform.capture_to_control_point(expected.x, expected.y)
    assert result.control_allowed is True
    assert result.control_x == pytest.approx(expected.x)
    assert result.control_y == pytest.approx(expected.y)
    assert result.control_x != pytest.approx(100 + result.roi_x)
    assert result.control_y != pytest.approx(50 + result.roi_y)


def test_latency_compensator_stops_when_compensated_point_leaves_roi() -> None:
    aim = AimPointGenerator().update(
        track=Track(track_id=30, cls=0, score=0.9, x1=600, y1=300, x2=640, y2=340),
        estimate=EstimatedTargetState(
            track_id=30,
            state_ts_ns=10_000_000,
            capture_ts_ns=8_000_000,
            x=620,
            y=320,
            vx=3000,
            vy=0,
            valid=True,
            prediction_confidence=1.0,
            velocity_measurements=5,
        ),
        config=AimPointConfig(ema_enabled=False),
    )
    estimate = EstimatedTargetState(
        track_id=30,
        state_ts_ns=10_000_000,
        capture_ts_ns=8_000_000,
        x=620,
        y=320,
        vx=3000,
        vy=0,
        valid=True,
        prediction_confidence=1.0,
        velocity_measurements=5,
    )

    result = LatencyCompensator().compensate(
        aim=aim,
        estimate=estimate,
        source_frame_id=70,
        compute_ts_ns=30_000_000,
        roi_offset_x=0,
        roi_offset_y=0,
        roi_width=640,
        roi_height=640,
        control_width=1920,
        control_height=1080,
        config=LatencyCompensationConfig(
            scale=1.0,
            max_compensation_ms=100,
            reject_if_age_exceeds_ms=100,
            max_compensation_px=100,
            min_velocity_px_s=0,
            max_velocity_px_s=4000,
            min_velocity_measurements=2,
            min_velocity_confidence=0,
            estimated_actuation_delay_ms=0,
        ),
    )

    assert result.control_allowed is False
    assert result.reason == "COMPENSATED_OUT_OF_ROI"


def test_latency_compensator_stops_when_control_point_leaves_control_frame() -> None:
    aim = AimPointGenerator().update(
        track=Track(track_id=31, cls=0, score=0.9, x1=580, y1=300, x2=620, y2=340),
        estimate=EstimatedTargetState(
            track_id=31,
            state_ts_ns=10_000_000,
            capture_ts_ns=8_000_000,
            x=600,
            y=320,
            vx=0,
            vy=0,
            valid=True,
        ),
        config=AimPointConfig(ema_enabled=False),
    )
    estimate = EstimatedTargetState(
        track_id=31,
        state_ts_ns=10_000_000,
        capture_ts_ns=8_000_000,
        x=600,
        y=320,
        vx=0,
        vy=0,
        valid=True,
    )

    result = LatencyCompensator().compensate(
        aim=aim,
        estimate=estimate,
        source_frame_id=71,
        compute_ts_ns=12_000_000,
        roi_offset_x=1500,
        roi_offset_y=0,
        roi_width=640,
        roi_height=640,
        control_width=1920,
        control_height=1080,
        config=LatencyCompensationConfig(enabled=False),
    )

    assert result.control_allowed is False
    assert result.reason == "COMPENSATED_OUT_OF_CONTROL"


def test_runtime_service_defensive_contract_for_broken_inference(tmp_path) -> None:
    """Any non-conforming inference input must keep the runtime silent.

    The runtime must never let a misbehaving inference (missing, broken,
    raising, returning None, or returning a malformed object) leak into
    the plugin pipeline. We verify the contract with one parametrized test
    rather than one test per failure mode.
    """
    scenarios: dict[str, object] = {
        "missing_inference": None,
        "broken_object": object(),
        "raising": type(
            "RaisingRuntime",
            (),
            {"infer": staticmethod(lambda frame: (_ for _ in ()).throw(RuntimeError("boom")))},
        )(),
        "none_result": type(
            "NoneResultRuntime",
            (),
            {"infer": staticmethod(lambda frame: None)},
        )(),
        "unavailable": type(
            "UnavailableRuntime",
            (),
            {"infer": staticmethod(lambda frame: InferenceResult(available=False, reason="nope"))},
        )(),
        "garbage_object": type(
            "GarbageRuntime",
            (),
            {"infer": staticmethod(lambda frame: object())},
        )(),
        "malformed_detection": type(
            "MalformedRuntime",
            (),
            {"infer": staticmethod(lambda frame: InferenceResult(available=True, detections=[object()]))},
        )(),
    }

    for name, inference in scenarios.items():
        service, kmnet = _runtime_service(tmp_path, inference)
        result = service.process_captured_frame(
            CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None)
        )
        assert result.control_intents == [], name
        assert result.execution_results == [], name
        assert kmnet.history == [], name
        assert service.last_inference_status["available"] is False, name


def test_runtime_infer_handles_engine_exceptions_and_invalid_results() -> None:
    class BrokenEngine:
        engine_id = "broken"

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"selected": self.engine_id, "available": True}

        def load(self, artifact_path, classes, input_shape) -> None:
            raise AssertionError("not used")

        def infer(self, frame) -> InferenceResult:
            raise RuntimeError("engine exploded")

    class NoneEngine:
        engine_id = "none"

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"selected": self.engine_id, "available": True}

        def load(self, artifact_path, classes, input_shape) -> None:
            raise AssertionError("not used")

        def infer(self, frame):
            return None

    broken_runtime = InferenceRuntime(BrokenEngine())
    none_runtime = InferenceRuntime(NoneEngine())

    broken_result = broken_runtime.infer(CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None))
    none_result = none_runtime.infer(CapturedFrame(7, 640, 480, "BGR", 123, 1.0, image=None))

    assert broken_result.available is False
    assert broken_result.reason == "engine exploded"

    assert none_result.available is False
    assert none_result.reason == "invalid inference result"
