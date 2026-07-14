from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from novasight.api import routes_model_ingress, routes_models
from novasight.api.routes_models import PublishRequest
from novasight.model_ingress import (
    EngineInspectionResult,
    ModelValidationReport,
    ModelStatus,
    ProfileValidation,
    TensorDescriptor,
    model_profile_validation_fingerprint,
)
from novasight.model_registry import ModelRegistry, inspect_model_artifact


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
                backend="deepstream_nvinfer",
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
    assert Path(inspected["profile_path"]).name == "player.engine.manifest.json"

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
    request, artifact, engine_path = _request(tmp_path)
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
    request.app.state.inference = SimpleNamespace(
        prepare_profile=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DeepStream diagnostics must not use the legacy TensorRT runtime")
        )
    )

    def deepstream_probe(profile, *, runtime, isolation):
        assert runtime is request.app.state.runtime
        with isolation:
            events.append("deepstream:nvinfer")
        validated = replace(
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
        return validated, ModelValidationReport(
            status="validated",
            engine_execution_ok=True,
            output_tensor_ok=True,
            decoder_ok=True,
            nms_ok=True,
            detection_batch_ok=True,
        )

    monkeypatch.setattr(
        routes_model_ingress,
        "probe_deepstream_model",
        deepstream_probe,
    )
    request.app.state.runtime = SimpleNamespace(
        pipeline=None,
        cancel_control=lambda reason: events.append(f"cancel:{reason}"),
    )

    response = routes_model_ingress.probe_engine_artifact(request, artifact.id)

    assert response["profile"]["status"] == ModelStatus.VALIDATED.value
    assert response["report"]["detection_batch_ok"] is True
    assert events[0] == "cancel:MODEL_DIAGNOSTIC"
    assert events[1:] == ["deepstream:nvinfer"]
    unified_manifest_path = routes_model_ingress.ModelProfileStore().path_for_engine(
        engine_path
    )
    unified_manifest = json.loads(unified_manifest_path.read_text(encoding="utf-8"))
    assert unified_manifest["model_profile"]["status"] == ModelStatus.VALIDATED.value
    assert unified_manifest["artifact"]["engine_path"] == engine_path.name
    assert not engine_path.with_name("model.manifest.json").exists()
    scanned = inspect_model_artifact(engine_path, force=True)
    assert scanned.status == "ready"
    assert scanned.manifest_path == unified_manifest_path

    class ActivationInference:
        def unload(self, _reason):
            events.append("unload")

        def status(self):
            return {"loaded": False, "selected": "deepstream_nvinfer"}

    request.app.state.inference = ActivationInference()
    project_id = request.app.state.models.get_version(artifact.version_id).project_id

    published = routes_models.publish(
        request,
        project_id,
        PublishRequest(artifact_id=artifact.id),
    )

    assert published["deployment"]["artifact_id"] == artifact.id
    assert published["inference"]["selected"] == "deepstream_nvinfer"
    assert events[-2:] == ["cancel:MODEL_SWITCH", "unload"]
    active_profile = routes_model_ingress.get_engine_profile(request, artifact.id)
    assert active_profile["profile"]["status"] == ModelStatus.ACTIVE.value

    profile_path = routes_model_ingress.ModelProfileStore().path_for_engine(engine_path)
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    raw_profile["model_profile"]["preprocess"]["color_format"] = "BGR"
    profile_path.write_text(json.dumps(raw_profile), encoding="utf-8")

    with pytest.raises(ValueError, match="not validated"):
        routes_model_ingress.load_validated_profile(engine_path)
