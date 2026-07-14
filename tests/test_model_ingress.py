from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from novasight.contracts import BBox
from novasight.inference.contracts import InferenceDetection, InferenceResult
from novasight.inference.input import PreparedTensorInput, TensorInputShape
from novasight.inference.jetson import JetsonGpuResourcePreprocessor
from novasight.inference.preprocess import prepare_host_tensor
from novasight.inference.preprocess import TensorPreprocessError
from novasight.inference.runtime import InferenceRuntime
from novasight.inference.tensorrt import TensorRtInferenceEngine
from novasight.model_ingress import (
    EngineInspectionErrorCode,
    EngineInspector,
    InferenceConfigBuilder,
    ModelProfileStore,
    ModelProbe,
    ModelProfileResolver,
    ModelStatus,
    PreprocessProfile,
    ProfileValidation,
    model_profile_validation_fingerprint,
)


class _FakeTensorIOMode:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class _FakeTensorRT:
    TensorIOMode = _FakeTensorIOMode


class _FakeEngine:
    name = "player-dynamic"
    num_io_tensors = 2
    num_optimization_profiles = 2

    def get_tensor_name(self, index: int) -> str:
        return ("images", "output0")[index]

    def get_tensor_mode(self, name: str) -> str:
        return _FakeTensorIOMode.INPUT if name == "images" else _FakeTensorIOMode.OUTPUT

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return (-1, 3, -1, -1) if name == "images" else (-1, 5, 8400)

    def get_tensor_dtype(self, _name: str) -> str:
        return "HALF"

    def get_tensor_format(self, _name: str) -> str:
        return "LINEAR"

    def get_tensor_bytes_per_component(self, _name: str) -> int:
        return 2

    def get_tensor_components_per_element(self, _name: str) -> int:
        return 1

    def get_tensor_vectorized_dim(self, _name: str) -> int:
        return -1

    def is_shape_inference_io(self, _name: str) -> bool:
        return False

    def get_tensor_profile_shape(
        self,
        name: str,
        profile_index: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert name == "images"
        if profile_index == 0:
            return (
                (1, 3, 320, 320),
                (1, 3, 640, 640),
                (1, 3, 640, 640),
            )
        return (
            (1, 3, 320, 512),
            (1, 3, 320, 512),
            (1, 3, 640, 1024),
        )


def test_engine_inspector_reads_authoritative_tensor_and_profile_contract(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspector = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    )

    result = inspector.inspect(engine_path)

    assert result.deserialize_ok is True
    assert result.compatible is True
    assert result.engine_name == "player-dynamic"
    assert result.has_dynamic_shape is True
    assert result.has_shape_input is False
    assert result.inputs[0].name == "images"
    assert result.inputs[0].engine_shape == (-1, 3, -1, -1)
    assert result.inputs[0].data_type == "float16"
    assert result.inputs[0].tensor_format == "linear"
    assert result.inputs[0].bytes_per_component == 2
    assert result.outputs[0].name == "output0"
    assert result.profiles[0].input_ranges["images"].optimum == (1, 3, 640, 640)
    assert result.profiles[1].input_ranges["images"].maximum == (1, 3, 640, 1024)


def test_engine_inspector_rejects_nhwc_instead_of_guessing_layout(tmp_path: Path) -> None:
    class NhwcEngine(_FakeEngine):
        num_optimization_profiles = 0

        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            return (1, 640, 640, 3) if name == "images" else (1, 8400, 5)

    engine_path = tmp_path / "nhwc.engine"
    engine_path.write_bytes(b"serialized-engine")

    result = EngineInspector(
        deserialize=lambda _path: (NhwcEngine(), _FakeTensorRT()),
    ).inspect(engine_path)

    assert result.deserialize_ok is True
    assert result.compatible is False
    assert result.error_code is EngineInspectionErrorCode.UNSUPPORTED_LAYOUT
    assert "NCHW" in result.raw_error


def test_profile_resolver_keeps_unknown_model_semantics_unconfirmed(tmp_path: Path) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)

    profile = ModelProfileResolver().resolve(
        engine_path=engine_path,
        inspection=inspection,
        display_name="Player dynamic",
    )

    assert profile.status is ModelStatus.NEEDS_CONFIGURATION
    assert profile.engine.sha256.startswith("sha256:")
    assert profile.engine.file_size == len(b"serialized-engine")
    assert profile.input.name == "images"
    assert profile.input.runtime_shape == (1, 3, 640, 640)
    assert profile.outputs[0].name == "output0"
    assert profile.outputs[0].shape == (1, 5, 8400)
    assert profile.outputs[0].engine_shape == (-1, 5, 8400)
    assert profile.preprocess.color_format == ""
    assert profile.decoder.parser_type == ""
    assert profile.parser_candidates[0].parser_type == "yolov8_raw"
    assert profile.parser_candidates[0].requires_confirmation is True


def test_profile_configuration_rejects_decoder_semantics_that_runtime_would_ignore(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)
    draft = ModelProfileResolver().resolve(
        engine_path=engine_path,
        inspection=inspection,
        display_name="Player",
    )

    with pytest.raises(ValueError, match="objectness"):
        ModelProfileResolver().configure(
            draft,
            color_format="RGB",
            scale=1.0 / 255.0,
            resize_mode="direct",
            parser_type="yolov8_raw",
            class_count=1,
            labels=["player"],
            bbox_format="xywh",
            has_objectness=True,
        )

    with pytest.raises(ValueError, match="three values"):
        ModelProfileResolver().configure(
            draft,
            color_format="RGB",
            scale=1.0 / 255.0,
            offsets=[0.0, 0.0],
            resize_mode="direct",
            parser_type="yolov8_raw",
            class_count=1,
            labels=["player"],
            bbox_format="xywh",
            has_objectness=False,
        )


def test_confirmed_profile_is_persisted_and_engine_change_invalidates_it(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)
    resolver = ModelProfileResolver()
    draft = resolver.resolve(
        engine_path=engine_path,
        inspection=inspection,
        display_name="Player dynamic",
    )

    confirmed = resolver.configure(
        draft,
        color_format="RGB",
        scale=1.0 / 255.0,
        resize_mode="direct",
        parser_type="yolov8_raw",
        class_count=1,
        labels=["player"],
        bbox_format="xywh",
        has_objectness=False,
    )
    store = ModelProfileStore()
    profile_path = store.write(confirmed)

    assert confirmed.status is ModelStatus.READY_FOR_PROBE
    assert profile_path.name == "player.engine.profile.json"
    assert store.load(profile_path).decoder.parser_type == "yolov8_raw"

    engine_path.write_bytes(b"replaced-engine")
    invalidated = store.load(profile_path)

    assert invalidated.status is ModelStatus.UNINSPECTED
    assert invalidated.validation.status == "not_run"
    assert invalidated.validation.issues == ("ENGINE_CONTENT_CHANGED",)


def test_model_profile_store_rejects_unknown_schema_version(tmp_path: Path) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)
    profile = ModelProfileResolver().resolve(
        engine_path=engine_path,
        inspection=inspection,
        display_name="Player",
    )
    store = ModelProfileStore()
    profile_path = store.write(profile)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 999
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        store.load(profile_path)


def test_model_probe_validates_detection_batch_without_committing_candidate(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)
    resolver = ModelProfileResolver()
    profile = resolver.configure(
        resolver.resolve(
            engine_path=engine_path,
            inspection=inspection,
            display_name="Player dynamic",
        ),
        color_format="RGB",
        scale=1.0 / 255.0,
        resize_mode="direct",
        parser_type="yolov8_raw",
        class_count=1,
        labels=["player"],
        bbox_format="xywh",
        has_objectness=False,
    )
    events: list[str] = []

    class Candidate:
        def infer(self, frame: object) -> InferenceResult:
            assert getattr(frame, "width") == 640
            assert getattr(frame, "height") == 640
            events.append("infer")
            return InferenceResult(
                available=True,
                classes=["player"],
                detections=[
                    InferenceDetection(
                        cls=0,
                        score=0.9,
                        box=BBox.from_xyxy(100, 120, 180, 260),
                    )
                ],
                debug={
                    "tensor_statistics": [
                        {
                            "name": "output0",
                            "shape": [1, 5, 8400],
                            "dtype": "float16",
                            "minimum": 0.0,
                            "maximum": 0.9,
                            "mean": 0.01,
                            "nan_count": 0,
                            "inf_count": 0,
                        }
                    ],
                    "decode": {
                        "raw_candidates": 8400,
                        "threshold_candidates": 1,
                        "nms_detections": 1,
                    },
                    "timings": {
                        "prepare_input_ms": 0.2,
                        "execute_total_ms": 1.5,
                        "total_ms": 2.0,
                    },
                },
            )

        def close(self) -> None:
            events.append("close")

    class Inference:
        def prepare_profile(self, candidate_profile, *, diagnostic: bool):
            assert candidate_profile is profile
            assert diagnostic is True
            events.append("prepare")
            return Candidate(), {"loaded": True, "warmed": True}

        def commit(self, *_args, **_kwargs):
            raise AssertionError("diagnostic probe must never commit a candidate")

    class Isolation:
        def __enter__(self):
            events.append("isolate")

        def __exit__(self, *_args):
            events.append("resume")

    validated, report = ModelProbe().run(
        profile,
        inference=Inference(),
        isolation=Isolation(),
        now_ns=lambda: 1_000_000_000,
    )

    assert validated.status is ModelStatus.VALIDATED
    assert report.status == "validated"
    assert report.engine_execution_ok is True
    assert report.decoder_ok is True
    assert report.nms_ok is True
    assert report.detection_batch_ok is True
    assert report.detection_batch is not None
    assert report.detection_batch.detections[0].cls == 0
    assert events == ["isolate", "prepare", "infer", "close", "resume"]


def test_inference_config_builder_derives_all_runtime_contracts_from_profile(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)
    resolver = ModelProfileResolver()
    profile = resolver.configure(
        resolver.resolve(
            engine_path=engine_path,
            inspection=inspection,
            display_name="Player dynamic",
        ),
        color_format="RGB",
        scale=1.0 / 255.0,
        resize_mode="direct",
        parser_type="yolov8_raw",
        class_count=1,
        labels=["player"],
        bbox_format="xywh",
        has_objectness=False,
    )
    profile = replace(
        profile,
        status=ModelStatus.VALIDATED,
        validation=ProfileValidation(
            status="validated",
            engine_execution_ok=True,
            decoder_ok=True,
            nms_ok=True,
            detection_batch_ok=True,
            profile_fingerprint=model_profile_validation_fingerprint(profile),
        ),
    )

    config = InferenceConfigBuilder().build(
        profile,
        parser_library_path=tmp_path / "libnovasight_parser.so",
    )

    assert config.input_shape == (1, 3, 640, 640)
    assert config.input_name == "images"
    assert config.output_names == ("output0",)
    assert config.preprocess.color_format == "RGB"
    assert config.decoder.runtime_decoder == "yolov8"
    assert config.decoder.class_count == 1
    assert "output-blob-names=output0" in config.deepstream_nvinfer
    assert "net-scale-factor=0.0039215686274509803" in config.deepstream_nvinfer


def test_runtime_config_requires_full_validation_and_supported_deepstream_preprocess(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)
    resolver = ModelProfileResolver()
    profile = resolver.configure(
        resolver.resolve(
            engine_path=engine_path,
            inspection=inspection,
            display_name="Player dynamic",
        ),
        color_format="RGB",
        scale=1.0 / 255.0,
        resize_mode="direct",
        parser_type="yolov8_raw",
        class_count=1,
        labels=["player"],
        bbox_format="xywh",
        has_objectness=False,
        mean=[0.5, 0.5, 0.5],
    )
    missing_nms = replace(
        profile,
        status=ModelStatus.VALIDATED,
        validation=ProfileValidation(
            status="validated",
            engine_execution_ok=True,
            decoder_ok=True,
            nms_ok=False,
            detection_batch_ok=True,
        ),
    )

    with pytest.raises(ValueError, match="validation report"):
        InferenceConfigBuilder().build(
            missing_nms,
            parser_library_path=tmp_path / "libnovasight_parser.so",
        )

    fully_validated = replace(
        missing_nms,
        validation=replace(
            missing_nms.validation,
            nms_ok=True,
            profile_fingerprint=model_profile_validation_fingerprint(profile),
        ),
    )
    with pytest.raises(ValueError, match="mean/std/offsets"):
        InferenceConfigBuilder().build(
            fully_validated,
            parser_library_path=tmp_path / "libnovasight_parser.so",
        )


def test_jetson_native_preprocess_rejects_model_semantics_it_cannot_execute() -> None:
    preprocessor = JetsonGpuResourcePreprocessor()
    preprocessor.configure_model_preprocess(
        PreprocessProfile(
            color_format="BGR",
            scale=1.0 / 255.0,
            resize_mode="direct",
        )
    )
    prepared = PreparedTensorInput(
        mode="gpu_buffer",
        buffer=object(),
        frame_id=1,
        capture_ts_ns=1,
        width=640,
        height=640,
        pixel_format="NV12",
        source_width=1920,
        source_height=1080,
        offset_x=640,
        offset_y=220,
        needs_resize=False,
        resource_kind="gstreamer_sample",
        resource_memory="nvmm",
        resource_source="appsink",
        dmabuf_fd=7,
        resource_metadata={},
        resource_width=640,
        resource_height=640,
        resource_pixel_format="NV12",
    )

    with pytest.raises(TensorPreprocessError, match="only supports RGB"):
        preprocessor.prepare(
            prepared,
            TensorInputShape(batch=1, channels=3, height=640, width=640),
        )


def test_inference_runtime_prepares_profile_candidate_without_replacing_active_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)
    resolver = ModelProfileResolver()
    profile = resolver.configure(
        resolver.resolve(
            engine_path=engine_path,
            inspection=inspection,
            display_name="Player dynamic",
        ),
        color_format="RGB",
        scale=1.0 / 255.0,
        resize_mode="direct",
        parser_type="yolov8_raw",
        class_count=1,
        labels=["player"],
        bbox_format="xywh",
        has_objectness=False,
    )
    events: list[object] = []

    class ActiveEngine:
        engine_id = "active"

        def available(self) -> bool:
            return True

        def last_reason(self) -> str:
            return ""

        def status(self) -> dict:
            return {"loaded": True}

    class Candidate:
        engine_id = "candidate"

        def configure_model_profile(self, value) -> None:
            assert value is profile
            events.append("configure")

        def set_diagnostic_mode(self, enabled: bool) -> None:
            events.append(("diagnostic", enabled))

        def load(self, path: Path, classes: list[str], input_shape: str) -> None:
            events.append(("load", path, classes, input_shape))

        def status(self) -> dict:
            return {"loaded": True, "warmed": True}

    runtime = InferenceRuntime(ActiveEngine())
    candidate = Candidate()
    monkeypatch.setattr(runtime, "_engine_for_artifact", lambda _path: candidate)

    prepared, status = runtime.prepare_profile(profile, diagnostic=True)

    assert runtime.engine.engine_id == "active"
    assert prepared is candidate
    assert status["loaded"] is True
    assert events == [
        "configure",
        ("diagnostic", True),
        ("load", engine_path.resolve(), ["player"], "1x3x640x640"),
    ]


def test_tensorrt_candidate_freezes_profile_semantics_before_loading(
    tmp_path: Path,
) -> None:
    engine_path = tmp_path / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    inspection = EngineInspector(
        deserialize=lambda _path: (_FakeEngine(), _FakeTensorRT()),
    ).inspect(engine_path)
    resolver = ModelProfileResolver()
    profile = resolver.configure(
        resolver.resolve(
            engine_path=engine_path,
            inspection=inspection,
            display_name="Player dynamic",
        ),
        color_format="BGR",
        scale=1.0 / 255.0,
        resize_mode="letterbox",
        parser_type="yolov8_raw",
        class_count=1,
        labels=["player"],
        bbox_format="xywh",
        has_objectness=False,
        symmetric_padding=True,
        padding_value=114.0,
    )
    candidate = TensorRtInferenceEngine()

    candidate.configure_model_profile(profile)
    candidate.set_diagnostic_mode(True)

    status = candidate.status()
    assert status["model_profile_id"] == profile.model_id
    assert status["parser"] == "yolov8_raw"
    assert status["runtime_decoder"] == "yolov8"
    assert status["preprocess"] == {
        "color_format": "BGR",
        "scale": 1.0 / 255.0,
        "offsets": [],
        "mean": [],
        "std": [],
        "resize_mode": "letterbox",
        "symmetric_padding": True,
        "padding_value": 114.0,
    }
    assert status["diagnostic"] is True


def test_host_preprocess_uses_profile_color_scale_offsets_mean_std_and_resize() -> None:
    import numpy as np

    prepared = PreparedTensorInput(
        mode="host_frame",
        buffer=np.asarray([[[10, 20, 30]]], dtype=np.uint8),
        frame_id=1,
        capture_ts_ns=1,
        width=1,
        height=1,
        pixel_format="BGR",
        source_width=1,
        source_height=1,
        offset_x=0,
        offset_y=0,
        needs_resize=False,
    )
    preprocess = PreprocessProfile(
        color_format="BGR",
        scale=0.5,
        offsets=(2.0, 4.0, 6.0),
        mean=(1.0, 2.0, 3.0),
        std=(1.0, 2.0, 4.0),
        resize_mode="direct",
    )

    result = prepare_host_tensor(
        prepared,
        TensorInputShape(batch=1, channels=3, height=1, width=1),
        model_preprocess=preprocess,
    )

    assert result.tensor.shape == (1, 3, 1, 1)
    assert result.tensor[0, :, 0, 0].tolist() == [3.0, 3.0, 2.25]
    assert result.metadata["resize_mode"] == "none"
    assert result.metadata["model_color_format"] == "BGR"
