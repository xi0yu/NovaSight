from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from novasight.api import routes_model_ingress, routes_models
from novasight.api.routes_models import PublishRequest
from novasight.contracts import BBox
from novasight.inference.contracts import InferenceDetection, InferenceResult
from novasight.model_ingress import (
    EngineInspectionResult,
    ModelStatus,
    TensorDescriptor,
)
from novasight.model_registry import ModelRegistry


def _inspection() -> EngineInspectionResult:
    return EngineInspectionResult(
        deserialize_ok=True,
        compatible=True,
        engine_name="player",
        inputs=(
            TensorDescriptor(
                name="images",
                io_mode="input",
                engine_shape=(1, 3, 640, 640),
                data_type="float16",
                tensor_format="linear",
                is_shape_tensor=False,
                bytes_per_component=2,
                components_per_element=1,
                vectorized_dim=-1,
            ),
        ),
        outputs=(
            TensorDescriptor(
                name="output0",
                io_mode="output",
                engine_shape=(1, 5, 8400),
                data_type="float16",
                tensor_format="linear",
                is_shape_tensor=False,
                bytes_per_component=2,
                components_per_element=1,
                vectorized_dim=-1,
            ),
        ),
    )


def _request(tmp_path: Path):
    registry = ModelRegistry(
        db_path=tmp_path / "novasight.db",
        data_dir=tmp_path / "models",
    )
    project = registry.create_project(name="player", description="")
    version = registry.create_version(
        project_id=project.id,
        version="v1",
        source_kind="onnx",
        source_path="player.onnx",
        classes=["target"],
        input_shape="engine-probe-required",
    )
    asset_dir = Path(registry.data_dir) / project.name / version.version
    asset_dir.mkdir(parents=True, exist_ok=True)
    engine_path = asset_dir / "player.engine"
    engine_path.write_bytes(b"serialized-engine")
    artifact = registry.create_artifact(
        version_id=version.id,
        kind="engine",
        path=engine_path.name,
        checksum="sha256:pending",
        status="pending",
    )
    state = SimpleNamespace(
        models=registry,
        inference=None,
        runtime=SimpleNamespace(pipeline=None, cancel_control=lambda _reason: None),
        config=SimpleNamespace(
            inference=SimpleNamespace(
                deepstream_parser_library="build/libnovasight_parser.so",
            )
        ),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state)), artifact, engine_path


def test_inspect_and_configure_model_profile_use_engine_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request, artifact, engine_path = _request(tmp_path)
    monkeypatch.setattr(
        routes_model_ingress,
        "EngineInspector",
        lambda: SimpleNamespace(inspect=lambda _path: _inspection()),
    )

    inspected = routes_model_ingress.inspect_engine_artifact(request, artifact.id)

    assert inspected["profile"]["status"] == ModelStatus.NEEDS_CONFIGURATION.value
    assert inspected["profile"]["input"]["runtime_shape"] == [1, 3, 640, 640]
    assert inspected["profile"]["outputs"][0]["name"] == "output0"
    assert Path(inspected["profile_path"]).name == "player.engine.profile.json"

    configured = routes_model_ingress.configure_engine_artifact(
        request,
        artifact.id,
        routes_model_ingress.ModelProfileConfigureRequest(
            color_format="RGB",
            scale=1.0 / 255.0,
            resize_mode="direct",
            parser_type="yolov8_raw",
            class_count=1,
            labels=["player"],
            bbox_format="xywh",
            has_objectness=False,
        ),
    )

    assert configured["profile"]["status"] == ModelStatus.READY_FOR_PROBE.value
    assert configured["profile"]["decoder"]["parser_type"] == "yolov8_raw"
    assert engine_path.is_file()


def test_probe_isolates_control_and_persists_validated_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request, artifact, _engine_path = _request(tmp_path)
    monkeypatch.setattr(
        routes_model_ingress,
        "EngineInspector",
        lambda: SimpleNamespace(inspect=lambda _path: _inspection()),
    )
    routes_model_ingress.inspect_engine_artifact(request, artifact.id)
    routes_model_ingress.configure_engine_artifact(
        request,
        artifact.id,
        routes_model_ingress.ModelProfileConfigureRequest(
            color_format="RGB",
            scale=1.0 / 255.0,
            resize_mode="direct",
            parser_type="yolov8_raw",
            class_count=1,
            labels=["player"],
            bbox_format="xywh",
            has_objectness=False,
        ),
    )
    events: list[str] = []

    class Candidate:
        def infer(self, _frame):
            events.append("infer")
            return InferenceResult(
                available=True,
                detections=[
                    InferenceDetection(
                        cls=0,
                        score=0.9,
                        box=BBox.from_xyxy(10, 10, 20, 20),
                    )
                ],
                classes=["player"],
                debug={
                    "tensor_statistics": [
                        {
                            "name": "output0",
                            "shape": [1, 5, 8400],
                            "dtype": "float16",
                            "minimum": 0,
                            "maximum": 1,
                            "mean": 0.1,
                            "nan_count": 0,
                            "inf_count": 0,
                        }
                    ],
                    "decode": {"nms_detections": 1},
                    "timings": {"execute_total_ms": 1.0},
                },
            )

        def close(self):
            events.append("close")

    request.app.state.inference = SimpleNamespace(
        prepare_profile=lambda _profile, diagnostic: (
            events.append(f"prepare:{diagnostic}") or Candidate(),
            {"loaded": True, "warmed": True},
        )
    )
    request.app.state.runtime = SimpleNamespace(
        pipeline=None,
        cancel_control=lambda reason: events.append(f"cancel:{reason}"),
    )

    response = routes_model_ingress.probe_engine_artifact(request, artifact.id)

    assert response["profile"]["status"] == ModelStatus.VALIDATED.value
    assert response["report"]["detection_batch_ok"] is True
    assert events[0] == "cancel:MODEL_DIAGNOSTIC"
    assert events[1:] == ["prepare:True", "infer", "close"]

    class ActiveCandidate:
        pass

    active_candidate = ActiveCandidate()

    class ActivationInference:
        def prepare_profile(self, profile, *, diagnostic: bool):
            assert profile.status is ModelStatus.VALIDATED
            assert diagnostic is False
            events.append("prepare:False")
            return active_candidate, {
                "loaded": True,
                "warmed": True,
                "classes": ["player"],
                "input_shape": "1x3x640x640",
            }

        def commit(self, candidate, **_kwargs):
            assert candidate is active_candidate
            events.append("commit")

        def status(self):
            return {"loaded": True, "warmed": True, "selected": "tensorrt"}

    request.app.state.inference = ActivationInference()
    project_id = request.app.state.models.get_version(artifact.version_id).project_id

    published = routes_models.publish(
        request,
        project_id,
        PublishRequest(artifact_id=artifact.id),
    )

    assert published["deployment"]["artifact_id"] == artifact.id
    assert published["inference"]["loaded"] is True
    assert events[-3:] == ["prepare:False", "cancel:MODEL_SWITCH", "commit"]
    active_profile = routes_model_ingress.get_engine_profile(request, artifact.id)
    assert active_profile["profile"]["status"] == ModelStatus.ACTIVE.value

    profile_path = routes_model_ingress.ModelProfileStore().path_for_engine(_engine_path)
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    raw_profile["preprocess"]["color_format"] = "BGR"
    profile_path.write_text(json.dumps(raw_profile), encoding="utf-8")

    with pytest.raises(ValueError, match="not validated"):
        routes_model_ingress.load_validated_profile(_engine_path)


def test_probe_latest_input_is_acquired_after_control_isolation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request, artifact, _engine_path = _request(tmp_path)
    monkeypatch.setattr(
        routes_model_ingress,
        "EngineInspector",
        lambda: SimpleNamespace(inspect=lambda _path: _inspection()),
    )
    routes_model_ingress.inspect_engine_artifact(request, artifact.id)
    routes_model_ingress.configure_engine_artifact(
        request,
        artifact.id,
        routes_model_ingress.ModelProfileConfigureRequest(
            color_format="RGB",
            scale=1.0 / 255.0,
            resize_mode="direct",
            parser_type="yolov8_raw",
            class_count=1,
            labels=["player"],
            bbox_format="xywh",
            has_objectness=False,
        ),
    )
    events: list[str] = []
    live_frame = SimpleNamespace(
        image=object(),
        width=640,
        height=640,
        pixel_format="RGB",
        frame_id=7,
        capture_ts_ns=123,
    )

    class Candidate:
        def infer(self, frame):
            assert frame is live_frame
            events.append("infer:latest")
            return InferenceResult(
                available=True,
                detections=[],
                classes=["player"],
                debug={
                    "tensor_statistics": [
                        {
                            "name": "output0",
                            "shape": [1, 5, 8400],
                            "dtype": "float16",
                            "minimum": 0,
                            "maximum": 0,
                            "mean": 0,
                            "nan_count": 0,
                            "inf_count": 0,
                        }
                    ],
                    "decode": {"nms_detections": 0},
                    "timings": {"execute_total_ms": 1.0},
                },
            )

        def close(self):
            events.append("close")

    request.app.state.inference = SimpleNamespace(
        prepare_profile=lambda _profile, diagnostic: (
            events.append(f"prepare:{diagnostic}") or Candidate(),
            {"loaded": True, "warmed": True},
        )
    )
    request.app.state.runtime = SimpleNamespace(
        pipeline=None,
        cancel_control=lambda reason: events.append(f"cancel:{reason}"),
    )
    request.app.state.capture = SimpleNamespace(
        get_latest_preview_frame=lambda: (
            events.append("capture:latest") or live_frame
        )
    )

    response = routes_model_ingress.probe_engine_artifact(
        request,
        artifact.id,
        routes_model_ingress.ModelProbeRequest(input_mode="latest"),
    )

    assert response["profile"]["status"] == ModelStatus.VALIDATED.value
    assert events == [
        "cancel:MODEL_DIAGNOSTIC",
        "prepare:True",
        "capture:latest",
        "infer:latest",
        "close",
    ]
