from __future__ import annotations

import hashlib
from types import SimpleNamespace

from fastapi.testclient import TestClient
import yaml
import pytest

from novasight.api import create_app
from novasight.api.deepstream_runtime import _configured_path
from novasight.capture.state import (
    CaptureCapability,
    CaptureCapabilities,
    CaptureRuntimeState,
)
from novasight.config import RuntimeConfig
from novasight.inference.jetson import JetsonGpuResourcePreprocessor
from novasight.license import TEST_MAX_LICENSE_KEY
from novasight.model_registry.deepstream_config import generate_nvinfer_config
from novasight.model_registry.manifest import (
    TensorSpec,
    build_engine_manifest,
    read_manifest,
    write_manifest,
)
from novasight.runtime import ControlFrameCsvRecorder, ControlFrameParquetRecorder


def _activate(client: TestClient) -> None:
    response = client.post("/api/license/activate", json={"key": TEST_MAX_LICENSE_KEY})
    assert response.status_code == 200


def test_runtime_start_deepstream_rejects_non_mjpeg_capture_format(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    cfg.capture.pixel_format = "NV12"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")
    registry.publish(project.id, artifact.id)

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "expects MJPEG capture" in response.json()["detail"]
    assert app.state.runtime.pipeline is None


def test_protected_api_requires_valid_license(tmp_path) -> None:
    client = TestClient(
        create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    )

    health = client.get("/healthz")
    schema = client.get("/api/config/schema")
    runtime = client.get("/api/runtime/state")

    assert health.status_code == 200
    assert schema.status_code == 200
    assert runtime.status_code == 401
    assert runtime.json()["detail"] == "license required"


def test_config_api_round_trips_strict_runtime_config(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)

    body = client.get("/api/config").json()
    body["capture"]["device"] = "/dev/video1"

    response = client.put("/api/config", json=body)

    assert response.status_code == 200
    assert response.json()["config"]["capture"]["device"] == "/dev/video1"
    assert app.state.runtime.config_store.status()["version"] == 1


def test_config_api_rejects_invalid_json_without_server_error(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)

    response = client.put(
        "/api/config",
        content=b"",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert "invalid JSON body" in response.json()["detail"]


def test_deepstream_configured_path_accepts_encoded_registry_paths(tmp_path) -> None:
    models_root = tmp_path / "data" / "models"
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(models=SimpleNamespace(data_dir=models_root))
        )
    )

    path = _configured_path(
        request,
        "data/models/combat_model%2Fdefault%2Fmodel.manifest.json",
    )

    assert path == (models_root / "combat_model/default/model.manifest.json").resolve(
        strict=False
    )


def test_config_api_clears_pipeline_when_deepstream_pipeline_setting_changes(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    cfg.inference.deepstream_io_mode = 2
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    stopped: list[str] = []
    app.state.runtime.pipeline = type(
        "Pipeline",
        (),
        {
            "running": False,
            "stop": lambda self: stopped.append("stopped"),
        },
    )()
    body = client.get("/api/config").json()
    body["inference"]["deepstream_io_mode"] = 4

    response = client.put("/api/config", json=body)

    assert response.status_code == 200
    assert stopped == ["stopped"]
    assert app.state.runtime.pipeline is None
    sections = response.json()["sections"]
    assert any(item["section"] == "runtime_pipeline" for item in sections)


def test_config_api_hot_updates_deepstream_postprocess_thresholds(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    cfg.inference.confidence_threshold = 0.25
    cfg.inference.nms_threshold = 0.45
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    updates: list[tuple[float, float]] = []

    class Source:
        def update_postprocess_thresholds(
            self,
            *,
            confidence_threshold: float,
            nms_threshold: float,
        ) -> None:
            updates.append((confidence_threshold, nms_threshold))

    class Pipeline:
        running = True
        detection_source = Source()

        def stop(self) -> None:
            pytest.fail("threshold-only config change must not rebuild DeepStream pipeline")

    app.state.runtime.pipeline = Pipeline()
    body = client.get("/api/config").json()
    body["inference"]["confidence_threshold"] = 0.6
    body["inference"]["nms_threshold"] = 0.3

    response = client.put("/api/config", json=body)

    assert response.status_code == 200
    assert app.state.runtime.pipeline is not None
    assert updates == [(0.6, 0.3)]
    sections = response.json()["sections"]
    assert not any(item["section"] == "runtime_pipeline" for item in sections)


def test_capture_select_clears_deepstream_pipeline_when_capture_profile_changes(tmp_path, monkeypatch) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    cfg.capture.device = "/dev/video0"
    cfg.capture.pixel_format = "MJPG"
    cfg.capture.width = 1920
    cfg.capture.height = 1080
    cfg.capture.fps = 120
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    stopped: list[str] = []
    app.state.runtime.pipeline = type(
        "Pipeline",
        (),
        {
            "running": True,
            "stop": lambda self: stopped.append("stopped"),
        },
    )()

    def configure_capture(*_args, **_kwargs):
        pytest.fail("DeepStream capture selection must not start the legacy capture session")

    def capabilities(device: str):
        return CaptureCapabilities(
            available=True,
            device=device,
            capabilities=[
                CaptureCapability(
                    pixel_format="MJPG",
                    width=1280,
                    height=720,
                    fps_list=[120],
                )
            ],
        )

    monkeypatch.setattr(app.state.capture, "configure", configure_capture)
    monkeypatch.setattr(app.state.capture, "capabilities", capabilities)

    response = client.post(
        "/api/capture/select",
        json={
            "device": "/dev/video1",
            "preference": "manual",
            "pixel_format": "MJPG",
            "width": 1280,
            "height": 720,
            "fps": 120,
        },
    )

    assert response.status_code == 200
    assert stopped == ["stopped"]
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False
    assert app.state.config.capture.device == "/dev/video1"
    assert app.state.capture.source is None
    assert app.state.capture.state.backend == "deepstream"
    sections = response.json()["report"]["sections"]
    assert any(item["section"] == "runtime_pipeline" for item in sections)
    assert response.json()["report"]["message"] == "DeepStream 采集配置已应用，未启动旧采集会话"


def test_image_source_clears_deepstream_pipeline(tmp_path, monkeypatch) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    stopped: list[str] = []
    app.state.runtime.pipeline = type(
        "Pipeline",
        (),
        {
            "running": True,
            "stop": lambda self: stopped.append("stopped"),
        },
    )()
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"not a real image; capture is mocked")

    def configure_image(path: str, *, fps: int):
        state = CaptureRuntimeState(available=True, device=path, backend="image")
        app.state.capture.state = state
        return state

    monkeypatch.setattr(app.state.capture, "configure_image", configure_image)

    response = client.post(
        "/api/capture/image",
        json={"path": str(image_path), "fps": 30},
    )

    assert response.status_code == 200
    assert stopped == ["stopped"]
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False
    assert app.state.config.source.default == "image"


def test_capture_stop_clears_stale_deepstream_pipeline(tmp_path, monkeypatch) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    stopped: list[str] = []
    app.state.runtime.pipeline = type(
        "Pipeline",
        (),
        {
            "running": False,
            "stop": lambda self: stopped.append("stopped"),
        },
    )()
    app.state.runtime.running = True

    def stop_capture(reason: str):
        state = CaptureRuntimeState(available=False, device="/dev/video0", last_error=reason)
        app.state.capture.state = state
        return state

    monkeypatch.setattr(app.state.capture, "stop", stop_capture)

    response = client.post("/api/capture/stop")

    assert response.status_code == 200
    assert stopped == ["stopped"]
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False
    assert app.state.config.source.default == "null"


def test_runtime_state_config_is_read_only_summary_not_editable_config(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)

    full_config = client.get("/api/config").json()
    state_config = client.get("/api/runtime/state").json()["config"]

    assert "control" in full_config
    assert "inference" in full_config
    assert "hardware" in full_config
    assert state_config["version"] == app.state.runtime.config_store.version
    assert state_config["capture"]["device"] == full_config["capture"]["device"]
    assert state_config["capture"]["memory"] == full_config["capture"]["memory"]
    assert state_config["roi"]["size"] == full_config["roi"]["size"]
    assert state_config["roi_size"] == full_config["roi"]["size"]
    assert state_config["consumers"] == full_config["consumers"]
    assert "control" not in state_config
    assert "inference" not in state_config
    assert "hardware" not in state_config
    assert "limits" not in state_config


def test_runtime_start_deepstream_requires_model_artifact_configuration(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "DeepStream backend requires an active model deployment" in response.json()["detail"]
    assert app.state.runtime.pipeline is None


def test_runtime_start_deepstream_rejects_non_single_batch_manifest(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "4x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"batch four engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [4, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [4, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")
    registry.publish(project.id, artifact.id)

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "single-source batch_size=1" in response.json()["detail"]
    assert app.state.runtime.pipeline is None


def test_runtime_start_deepstream_rejects_stale_nvinfer_config(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    (model_dir / "deepstream.ini").write_text("[property]\n", encoding="utf-8")
    registry.publish(project.id, artifact.id)

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "nvinfer config does not match model manifest" in response.json()["detail"]
    assert app.state.runtime.pipeline is None


def test_runtime_start_deepstream_rejects_engine_changed_after_prepare(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")
    registry.publish(project.id, artifact.id)
    engine_path.write_bytes(b"different engine bytes")

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "engine file" in response.json()["detail"]
    assert app.state.runtime.pipeline is None


def test_runtime_start_deepstream_rejects_nvinfer_config_with_wrong_engine_path(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    wrong_engine_path = model_dir / "old.engine"
    wrong_engine_path.write_bytes(b"old engine")
    (model_dir / "deepstream.ini").write_text(
        config_text.replace(str(engine_path.resolve()), str(wrong_engine_path.resolve())),
        encoding="utf-8",
    )
    registry.publish(project.id, artifact.id)

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "model-engine-file does not match artifact" in response.json()["detail"]
    assert app.state.runtime.pipeline is None


def test_runtime_start_deepstream_clears_pipeline_when_source_start_fails(tmp_path, monkeypatch) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    cfg.inference.deepstream_io_mode = 4
    cfg.inference.deepstream_batched_push_timeout_us = 12000
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")
    registry.publish(project.id, artifact.id)
    stopped: list[str] = []
    constructed: dict[str, object] = {}

    class FailingDeepStreamBackend:
        roi_width = 480
        roi_height = 480

        def __init__(self, **kwargs) -> None:
            constructed.update(kwargs)
            return None

        def start(self) -> None:
            raise RuntimeError("DeepStream pipeline error: nvinfer failed")

        def stop(self) -> None:
            stopped.append("stopped")

        def status(self) -> dict[str, object]:
            return {
                "selected": "deepstream",
                "available": True,
                "running": False,
                "last_error": "DeepStream pipeline error: nvinfer failed",
            }

    monkeypatch.setattr(
        "novasight.api.deepstream_runtime.DeepStreamDetectionBackend",
        FailingDeepStreamBackend,
    )

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "nvinfer failed" in response.json()["detail"]
    assert stopped == ["stopped"]
    pipeline_config = constructed["pipeline_config"]
    assert pipeline_config.io_mode == 4
    assert pipeline_config.batched_push_timeout_us == 12000
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False


def test_runtime_start_deepstream_clears_existing_pipeline_when_configuration_fails(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    stopped: list[str] = []

    class ExistingPipeline:
        def start(self) -> None:
            raise AssertionError("old pipeline must not be restarted")

        def stop(self) -> None:
            stopped.append("stopped")

    app.state.runtime.pipeline = ExistingPipeline()
    app.state.runtime.running = True

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "DeepStream backend requires an active model deployment" in response.json()["detail"]
    assert stopped == ["stopped"]
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False


def test_runtime_start_deepstream_rejects_configured_paths_outside_model_registry(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    cfg.inference.deepstream_manifest_path = "../outside/model.manifest.json"
    cfg.inference.deepstream_config_path = "combat/default/deepstream.ini"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert "must stay inside model registry" in response.json()["detail"]
    assert app.state.runtime.pipeline is None


def test_device_capabilities_describe_jetson_runtime_boundary(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    cfg.inference.deepstream_manifest_path = "combat/default/model.manifest.json"
    cfg.inference.deepstream_config_path = "combat/default/deepstream.ini"
    cfg.inference.deepstream_io_mode = 4
    cfg.inference.deepstream_batched_push_timeout_us = 12000
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)

    unauthorized = client.get("/api/device/capabilities")
    _activate(client)
    response = client.get("/api/device/capabilities")

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    body = response.json()
    assert body["node_role"] == "jetson_runtime"
    assert body["studio_role"] == "remote_manager"
    assert body["realtime_owner"] == "jetson"
    assert body["critical_path"] == {
        "host_in_realtime_loop": False,
        "jetson_runs_capture": True,
        "jetson_runs_inference": True,
        "jetson_runs_control": True,
    }
    assert body["transports"]["rest"] is True
    assert body["transports"]["websocket"] is True
    assert body["transports"]["grpc"] is False
    assert body["endpoints"]["status_ws"] == "/ws/status"
    assert body["endpoints"]["capture_preview"] == "/api/capture/stream.mjpg"
    assert body["endpoints"]["model_scan"] == "/api/models/scan"
    assert body["features"]["model_management"] is True
    assert body["features"]["remote_runtime_control"] is True
    assert body["features"]["deepstream_pipeline_generation"] is True
    assert body["features"]["deepstream_runtime_backend"] == body["deepstream"]["available"]
    assert body["features"]["deepstream_realtime_path_selected"] is True
    assert body["features"]["legacy_nvmm_capture_configured"] is True
    assert body["features"]["nvmm_capture_configured"] is True
    assert body["active_config"]["inference_backend"] == "deepstream"
    assert body["active_config"]["deepstream_manifest_path"] == (
        "combat/default/model.manifest.json"
    )
    assert body["active_config"]["deepstream_config_path"] == "combat/default/deepstream.ini"
    assert body["active_config"]["deepstream_io_mode"] == 4
    assert body["active_config"]["deepstream_batched_push_timeout_us"] == 12000
    assert set(body["deepstream"]) == {"available", "reason", "detail"}


def test_model_scan_api_reports_manifest_configuration_status(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    models_dir = registry.data_dir
    ready_dir = models_dir / "ready_model" / "default"
    ready_dir.mkdir(parents=True)
    ready_engine = ready_dir / "model.engine"
    ready_engine.write_bytes(b"ready")
    manifest = build_engine_manifest(
        model_id="ready_model",
        display_name="Ready Model",
        engine_path=ready_engine,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, ready_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=ready_engine)
    (ready_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")
    missing_config_dir = models_dir / "missing_config" / "default"
    missing_config_dir.mkdir(parents=True)
    missing_config_engine = missing_config_dir / "model.engine"
    missing_config_engine.write_bytes(b"missing config")
    missing_config_manifest = build_engine_manifest(
        model_id="missing_config",
        display_name="Missing Config",
        engine_path=missing_config_engine,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(missing_config_manifest, missing_config_dir / "model.manifest.json")
    need_confirm_dir = models_dir / "need_confirm" / "default"
    need_confirm_dir.mkdir(parents=True)
    (need_confirm_dir / "model.engine").write_bytes(b"needs manifest")

    response = client.get("/api/models/scan")

    assert response.status_code == 200
    body = response.json()
    statuses = {item["path"]: item for item in body["artifacts"]}
    assert statuses["ready_model/default/model.engine"]["status"] == "ready"
    assert statuses["ready_model/default/model.engine"]["model_fingerprint"] == manifest.model_fingerprint
    assert statuses["ready_model/default/model.engine"]["deepstream_config_path"] == "ready_model/default/deepstream.ini"
    assert statuses["missing_config/default/model.engine"]["status"] == "need_confirm"
    assert statuses["missing_config/default/model.engine"]["reason"] == "DeepStream nvinfer config is missing"
    assert statuses["need_confirm/default/model.engine"]["status"] == "need_confirm"
    assert statuses["need_confirm/default/model.engine"]["reason"] == "model manifest is missing"
    assert body["artifact_status_counts"]["ready"] == 1
    assert body["artifact_status_counts"]["need_confirm"] == 2

    missing_project = next(
        project for project in registry.list_projects() if project.name == "missing_config"
    )
    missing_version = registry.list_versions(missing_project.id)[0]
    missing_artifact = registry.list_artifacts(missing_version.id)[0]
    assert missing_artifact.status == "pending"
    ready_project = next(project for project in registry.list_projects() if project.name == "ready_model")
    ready_version = registry.list_versions(ready_project.id)[0]
    ready_artifact = registry.list_artifacts(ready_version.id)[0]
    assert ready_artifact.status == "ready"


def test_model_upload_engine_requires_deepstream_prepare_and_updates_checksum(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)

    first = client.post(
        "/api/models/upload",
        data={
            "project_name": "combat_model",
            "version": "default",
            "description": "",
            "classes": "body,head",
            "input_shape": "1x3x256x256",
        },
        files={"file": ("model.engine", b"engine-v1", "application/octet-stream")},
    )

    assert first.status_code == 200
    first_artifact = first.json()["artifact"]
    assert first_artifact["status"] == "pending"
    assert first_artifact["checksum"] == f"sha256:{hashlib.sha256(b'engine-v1').hexdigest()}"

    second = client.post(
        "/api/models/upload",
        data={
            "project_name": "combat_model",
            "version": "default",
            "description": "",
            "classes": "body,head",
            "input_shape": "1x3x256x256",
        },
        files={"file": ("model.engine", b"engine-v2", "application/octet-stream")},
    )

    assert second.status_code == 200
    second_artifact = second.json()["artifact"]
    assert second_artifact["id"] == first_artifact["id"]
    assert second_artifact["status"] == "pending"
    assert second_artifact["checksum"] == f"sha256:{hashlib.sha256(b'engine-v2').hexdigest()}"


def test_model_deepstream_prepare_api_writes_manifest_and_nvinfer_config(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "pending",
    )

    response = client.post(
        f"/api/models/artifacts/{artifact.id}/deepstream/prepare",
        json={
            "model_id": "combat_model",
            "display_name": "Combat Model",
            "runtime_precision": "fp32",
            "input_name": "images",
            "input_shape": [1, 3, 256, 256],
            "input_dtype": "float16",
            "input_color_format": "BGR",
            "input_scale_factor": 1.0,
            "maintain_aspect_ratio": True,
            "symmetric_padding": True,
            "output_name": "output0",
            "output_shape": [1, 8, 1344],
            "output_dtype": "float32",
            "class_count": 4,
            "confidence_threshold": 0.35,
            "nms_iou_threshold": 0.5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["artifact"]["id"] == artifact.id
    assert body["artifact"]["status"] == "ready"
    assert body["manifest_path"] == "combat_model/default/model.manifest.json"
    assert body["deepstream_config_path"] == "combat_model/default/deepstream.ini"
    assert registry.get_artifact(artifact.id).status == "ready"
    manifest = read_manifest(model_dir / "model.manifest.json")
    assert manifest.model_id == "combat_model"
    assert manifest.display_name == "Combat Model"
    assert manifest.input.shape == [1, 3, 256, 256]
    assert manifest.input.dtype == "float16"
    assert manifest.runtime.precision == "fp32"
    assert manifest.input.color_format == "BGR"
    assert manifest.input.scale_factor == pytest.approx(1.0)
    assert manifest.input.maintain_aspect_ratio is True
    assert manifest.input.symmetric_padding is True
    assert manifest.output.shape == [1, 8, 1344]
    assert manifest.output.dtype == "float32"
    assert manifest.output.class_count == 4
    assert manifest.output.class_names == ["body", "head", "team", "bot"]
    assert manifest.postprocess.confidence_threshold == 0.35
    deepstream_ini = (model_dir / "deepstream.ini").read_text(encoding="utf-8")
    assert "# novasight-config-fingerprint=nvinfer-template-v1:" in deepstream_ini
    assert f"model-engine-file={engine_path.resolve()}" in deepstream_ini
    assert "network-type=100" in deepstream_ini
    assert "network-mode=0" in deepstream_ini
    assert "output-tensor-meta=1" in deepstream_ini
    assert "output-blob-names=output0" in deepstream_ini
    assert "model-color-format=1" in deepstream_ini
    assert "net-scale-factor=1" in deepstream_ini
    assert "maintain-aspect-ratio=1" in deepstream_ini
    assert "symmetric-padding=1" in deepstream_ini


def test_model_deepstream_prepare_api_accepts_yolov5_objectness_channels(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x320x320",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "pending",
    )

    response = client.post(
        f"/api/models/artifacts/{artifact.id}/deepstream/prepare",
        json={
            "model_id": "combat_model",
            "display_name": "Combat Model",
            "runtime_precision": "fp16",
            "input_name": "images",
            "input_shape": [1, 3, 320, 320],
            "input_dtype": "float32",
            "input_color_format": "RGB",
            "input_scale_factor": 0.00392156862745098,
            "maintain_aspect_ratio": False,
            "symmetric_padding": False,
            "output_name": "output0",
            "output_shape": [1, 9, 6300],
            "output_dtype": "float32",
            "class_count": 4,
            "confidence_threshold": 0.25,
            "nms_iou_threshold": 0.45,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["manifest_path"] == "combat_model/default/model.manifest.json"
    assert body["deepstream_config_path"] == "combat_model/default/deepstream.ini"
    manifest = read_manifest(model_dir / "model.manifest.json")
    assert manifest.input.shape == [1, 3, 320, 320]
    assert manifest.output.shape == [1, 9, 6300]
    assert manifest.output.class_count == 4
    assert manifest.output.class_names == ["body", "head", "team", "bot"]


def test_model_deepstream_prepare_api_rejects_class_count_channel_mismatch(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )

    response = client.post(
        f"/api/models/artifacts/{artifact.id}/deepstream/prepare",
        json={
            "model_id": "combat_model",
            "display_name": "Combat Model",
            "input_name": "images",
            "input_shape": [1, 3, 256, 256],
            "output_name": "output0",
            "output_shape": [1, 8, 1344],
            "class_count": 1,
        },
    )

    assert response.status_code == 400
    assert "channels must equal 4 + class_count or 5 + class_count" in response.json()["detail"]
    assert not (model_dir / "model.manifest.json").exists()
    assert not (model_dir / "deepstream.ini").exists()


def test_model_deepstream_prepare_api_rejects_class_count_definition_mismatch(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "pending",
    )

    response = client.post(
        f"/api/models/artifacts/{artifact.id}/deepstream/prepare",
        json={
            "model_id": "combat_model",
            "display_name": "Combat Model",
            "input_name": "images",
            "input_shape": [1, 3, 256, 256],
            "output_name": "output0",
            "output_shape": [1, 8, 1344],
            "class_count": 4,
        },
    )

    assert response.status_code == 400
    assert "class_count must match the model version class definitions" in response.json()["detail"]
    assert not (model_dir / "model.manifest.json").exists()
    assert not (model_dir / "deepstream.ini").exists()


def test_model_deepstream_pipeline_api_uses_confirmed_manifest_and_config(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")

    response = client.post(
        f"/api/models/artifacts/{artifact.id}/deepstream/pipeline",
        json={
            "device": "/dev/video0",
            "capture_width": 1920,
            "capture_height": 1080,
            "fps": 120,
            "roi_left": 720,
            "roi_top": 300,
            "roi_size": 480,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["manifest_path"] == "combat_model/default/model.manifest.json"
    assert body["deepstream_config_path"] == "combat_model/default/deepstream.ini"
    assert body["model_input"]["width"] == 256
    assert body["model_input"]["height"] == 256
    assert body["roi"] == {
        "left": 720,
        "top": 300,
        "right": 1200,
        "bottom": 780,
        "size": 480,
    }
    assert body["capture"]["pixel_format"] == "MJPG"
    pipeline = body["pipeline"]
    assert "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=256,height=256" in pipeline
    assert "nvstreammux name=mux batch-size=1 width=256 height=256 live-source=1 sync-inputs=0" in pipeline
    assert f"nvinfer name=primary-infer config-file-path={(model_dir / 'deepstream.ini').resolve()}" in pipeline
    assert "videoconvert" not in pipeline


def test_model_deepstream_pipeline_api_rejects_non_mjpeg_capture_format(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")

    response = client.post(
        f"/api/models/artifacts/{artifact.id}/deepstream/pipeline",
        json={
            "device": "/dev/video0",
            "pixel_format": "NV12",
            "capture_width": 1920,
            "capture_height": 1080,
            "fps": 120,
            "roi_left": 720,
            "roi_top": 300,
            "roi_size": 480,
        },
    )

    assert response.status_code == 400
    assert "expects MJPEG capture" in response.json()["detail"]


def test_model_deepstream_pipeline_api_rejects_stale_config(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    (model_dir / "deepstream.ini").write_text("[property]\n", encoding="utf-8")

    response = client.post(
        f"/api/models/artifacts/{artifact.id}/deepstream/pipeline",
        json={
            "device": "/dev/video0",
            "capture_width": 1920,
            "capture_height": 1080,
            "fps": 120,
            "roi_left": 720,
            "roi_top": 300,
            "roi_size": 480,
        },
    )

    assert response.status_code == 400
    assert "nvinfer config does not match model manifest" in response.json()["detail"]


def test_model_deepstream_pipeline_api_rejects_config_property_mismatch(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    tampered_config = config_text.replace("output-blob-names=output0", "output-blob-names=boxes")
    (model_dir / "deepstream.ini").write_text(tampered_config, encoding="utf-8")

    response = client.post(
        f"/api/models/artifacts/{artifact.id}/deepstream/pipeline",
        json={
            "device": "/dev/video0",
            "capture_width": 1920,
            "capture_height": 1080,
            "fps": 120,
            "roi_left": 720,
            "roi_top": 300,
            "roi_size": 480,
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "nvinfer config properties do not match model manifest" in detail
    assert "output-blob-names=boxes expected output0" in detail


def test_model_publish_deepstream_keeps_legacy_runtime_out_of_switch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "novasight.runtime.service.check_deepstream_dependencies",
        lambda: {"available": True, "reason": "", "detail": ""},
    )
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(
        app.state.inference,
        "prepare",
        lambda *_args, **_kwargs: pytest.fail("legacy inference prepare should not run"),
    )
    monkeypatch.setattr(
        app.state.inference,
        "commit",
        lambda *_args, **_kwargs: pytest.fail("legacy inference commit should not run"),
    )

    class Pipeline:
        running = False

        def __init__(self) -> None:
            self.stop_count = 0

        def stop(self) -> None:
            self.stop_count += 1
            self.running = False

    pipeline = Pipeline()
    app.state.runtime.pipeline = pipeline
    app.state.runtime.running = True

    response = client.post(
        f"/api/models/projects/{project.id}/publish",
        json={"artifact_id": artifact.id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["deployment"]["artifact_id"] == artifact.id
    assert body["inference"]["selected"] == "deepstream"
    assert body["inference"]["loaded"] is False
    assert body["report"]["applied"] is True
    assert body["report"]["backend"] == "deepstream"
    assert body["report"]["sections"][1]["status"] == "applied"
    assert pipeline.stop_count == 1
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False
    assert (
        app.state.config.inference.deepstream_manifest_path
        == "combat_model/default/model.manifest.json"
    )
    assert app.state.config.inference.deepstream_config_path == "combat_model/default/deepstream.ini"
    assert (
        app.state.runtime.config.inference.deepstream_manifest_path
        == "combat_model/default/model.manifest.json"
    )
    assert (
        app.state.runtime.config.inference.deepstream_config_path
        == "combat_model/default/deepstream.ini"
    )
    persisted = yaml.safe_load((tmp_path / "missing.yaml").read_text(encoding="utf-8"))
    assert persisted["inference"]["deepstream_manifest_path"] == (
        "combat_model/default/model.manifest.json"
    )
    assert persisted["inference"]["deepstream_config_path"] == (
        "combat_model/default/deepstream.ini"
    )
    runtime_state = client.get("/api/runtime/state").json()
    assert runtime_state["active_model"]["artifact"]["id"] == artifact.id
    assert runtime_state["inference"]["selected"] == "deepstream"
    assert runtime_state["inference"]["configured"] is True
    assert runtime_state["inference"]["loaded"] is False
    assert "nvinfer loads" in runtime_state["inference"]["reason"]


def test_model_publish_deepstream_resumes_running_pipeline_with_new_source(tmp_path, monkeypatch) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")

    class OldPipeline:
        running = True

        def __init__(self) -> None:
            self.stop_count = 0

        def stop(self) -> None:
            self.stop_count += 1
            self.running = False

    class FakeDeepStreamSource:
        roi_width = 480
        roi_height = 480

        def __init__(self) -> None:
            self.running = False
            self.start_count = 0
            self.stop_count = 0

        def start(self) -> None:
            self.running = True
            self.start_count += 1

        def stop(self) -> None:
            self.running = False
            self.stop_count += 1

        def latest_result(self, *, after_frame_id=None):
            del after_frame_id
            return None

        def status(self) -> dict[str, object]:
            return {
                "selected": "deepstream",
                "available": True,
                "running": self.running,
                "last_error": "",
            }

    old_pipeline = OldPipeline()
    fake_source = FakeDeepStreamSource()
    app.state.runtime.pipeline = old_pipeline
    app.state.runtime.running = True
    monkeypatch.setattr(
        "novasight.api.routes_models.build_deepstream_detection_source",
        lambda _request: fake_source,
    )

    response = client.post(
        f"/api/models/projects/{project.id}/publish",
        json={"artifact_id": artifact.id},
    )

    try:
        assert response.status_code == 200
        body = response.json()
        assert old_pipeline.stop_count >= 1
        assert fake_source.start_count == 1
        assert body["inference"]["selected"] == "deepstream"
        assert body["inference"]["loaded"] is True
        assert body["inference"]["running"] is True
        assert body["inference"]["reason"] == "DeepStream runtime resumed after model switch"
        assert body["report"]["sections"][1]["status"] == "applied"
        assert app.state.runtime.pipeline is not old_pipeline
        assert app.state.runtime.pipeline.running is True
    finally:
        if app.state.runtime.pipeline is not None:
            app.state.runtime.pipeline.stop()


def test_model_publish_deepstream_cleans_failed_resume_pipeline(tmp_path, monkeypatch) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")

    class OldPipeline:
        running = True

        def stop(self) -> None:
            self.running = False

    class FakeDeepStreamSource:
        def status(self) -> dict[str, object]:
            return {"selected": "deepstream", "available": True, "running": False}

    cleanup: list[str] = []

    class FailingRuntimePipeline:
        def __init__(self, **_kwargs) -> None:
            self.running = False

        def start(self) -> None:
            self.running = True
            raise RuntimeError("synthetic DeepStream resume failure")

        def stop(self) -> None:
            cleanup.append("stopped")
            self.running = False

    app.state.runtime.pipeline = OldPipeline()
    app.state.runtime.running = True
    monkeypatch.setattr(
        "novasight.api.routes_models.build_deepstream_detection_source",
        lambda _request: FakeDeepStreamSource(),
    )
    monkeypatch.setattr(
        "novasight.api.routes_models.RuntimePipeline",
        FailingRuntimePipeline,
    )

    response = client.post(
        f"/api/models/projects/{project.id}/publish",
        json={"artifact_id": artifact.id},
    )

    assert response.status_code == 200
    body = response.json()
    assert cleanup == ["stopped"]
    assert body["inference"]["resume_failed"] is True
    assert "synthetic DeepStream resume failure" in body["inference"]["reason"]
    assert body["report"]["applied"] is False
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False


def test_model_publish_deepstream_rejects_stale_config(tmp_path, monkeypatch) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    (model_dir / "deepstream.ini").write_text("[property]\n", encoding="utf-8")
    monkeypatch.setattr(
        app.state.inference,
        "prepare",
        lambda *_args, **_kwargs: pytest.fail("legacy inference prepare should not run"),
    )

    response = client.post(
        f"/api/models/projects/{project.id}/publish",
        json={"artifact_id": artifact.id},
    )

    assert response.status_code == 400
    assert "nvinfer config does not match model manifest" in response.json()["detail"]
    assert registry.get_active_deployment() is None
    assert app.state.runtime.pipeline is None


def test_model_publish_deepstream_rejects_engine_changed_after_prepare(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    version = registry.create_version(
        project.id,
        "default",
        "onnx",
        "model.engine",
        ["body", "head", "team", "bot"],
        "1x3x256x256",
    )
    model_dir = registry.data_dir / project.name / version.version
    model_dir.mkdir(parents=True, exist_ok=True)
    engine_path = model_dir / "model.engine"
    engine_path.write_bytes(b"confirmed engine")
    artifact = registry.create_artifact(
        version.id,
        "engine",
        "model.engine",
        "sha256:test",
        "ready",
    )
    manifest = build_engine_manifest(
        model_id="combat_model",
        display_name="Combat Model",
        engine_path=engine_path,
        input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
        output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
        class_count=4,
        validated=True,
    )
    write_manifest(manifest, model_dir / "model.manifest.json")
    config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
    (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")
    engine_path.write_bytes(b"different engine bytes")

    response = client.post(
        f"/api/models/projects/{project.id}/publish",
        json={"artifact_id": artifact.id},
    )

    assert response.status_code == 400
    assert "engine file" in response.json()["detail"]
    assert registry.get_active_deployment() is None


def test_model_rollback_deepstream_rejects_stale_previous_config_without_changing_deployment(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    registry = app.state.models
    project = registry.create_project("combat_model", "")
    artifact_ids: list[int] = []
    model_dirs = []
    for version_name, engine_bytes in [
        ("v1", b"engine one"),
        ("v2", b"engine two"),
    ]:
        version = registry.create_version(
            project.id,
            version_name,
            "onnx",
            "model.engine",
            ["body", "head", "team", "bot"],
            "1x3x256x256",
        )
        model_dir = registry.data_dir / project.name / version.version
        model_dir.mkdir(parents=True, exist_ok=True)
        engine_path = model_dir / "model.engine"
        engine_path.write_bytes(engine_bytes)
        artifact = registry.create_artifact(
            version.id,
            "engine",
            "model.engine",
            f"sha256:{version_name}",
            "ready",
        )
        manifest = build_engine_manifest(
            model_id=f"combat_model_{version_name}",
            display_name=f"Combat Model {version_name}",
            engine_path=engine_path,
            input_spec=TensorSpec("images", [1, 3, 256, 256], "float32", "NCHW"),
            output_spec=TensorSpec("output0", [1, 8, 1344], "float32", "NCHW"),
            class_count=4,
            validated=True,
        )
        write_manifest(manifest, model_dir / "model.manifest.json")
        config_text, _fingerprint = generate_nvinfer_config(manifest, engine_path=engine_path)
        (model_dir / "deepstream.ini").write_text(config_text, encoding="utf-8")
        artifact_ids.append(artifact.id)
        model_dirs.append(model_dir)

    first_artifact_id, second_artifact_id = artifact_ids
    client.post(
        f"/api/models/projects/{project.id}/publish",
        json={"artifact_id": first_artifact_id},
    )
    client.post(
        f"/api/models/projects/{project.id}/publish",
        json={"artifact_id": second_artifact_id},
    )
    model_dirs[0].joinpath("deepstream.ini").write_text("[property]\n", encoding="utf-8")

    response = client.post(f"/api/models/projects/{project.id}/rollback")

    assert response.status_code == 400
    assert "nvinfer config does not match model manifest" in response.json()["detail"]
    deployment = registry.get_active_deployment()
    assert deployment is not None
    assert deployment.artifact_id == second_artifact_id
    assert deployment.previous_artifact_id == first_artifact_id


def test_config_schema_matches_runtime_config_and_update_syncs_runtime_objects(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)

    schema = client.get("/api/config/schema").json()
    body = schema["values"]
    body["control"]["output_mode"] = "kmnet"
    body["capture"]["device"] = "/dev/video7"
    body["capture"]["memory"] = "nvmm"
    body["roi"]["size"] = 320

    response = client.put("/api/config", json=body)

    assert response.status_code == 200
    assert response.json()["config"]["capture"]["device"] == "/dev/video7"
    assert response.json()["config"]["capture"]["memory"] == "nvmm"
    assert app.state.capture.config.device == "/dev/video7"
    assert app.state.capture.config.memory == "nvmm"
    assert app.state.capture.roi_size == 320
    assert app.state.runtime.config.capture.device == "/dev/video7"
    assert app.state.runtime.config.capture.memory == "nvmm"
    assert isinstance(app.state.inference._gpu_preprocessor, JetsonGpuResourcePreprocessor)
    runtime_state = client.get("/api/runtime/state").json()
    bridge_status = runtime_state["inference"]["gpu_preprocessor"]
    assert bridge_status["selected"] == "jetson_nvmm_cuda"
    assert bridge_status["enabled"] is True
    assert bridge_status["available"] is False
    assert bridge_status["reason"] == "JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE"
    assert app.state.runtime.executors.selected == "kmnet"
    assert any(section["id"] == "hardware" for section in schema["sections"])


def test_config_schema_exposes_image_source_and_consumers(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)

    schema = client.get("/api/config/schema").json()
    body = schema["values"]

    section_ids = {section["id"] for section in schema["sections"]}
    field_paths = {
        field["path"]
        for section in schema["sections"]
        for field in section["fields"]
    }

    assert "source" in section_ids
    assert "consumers" in section_ids
    assert body["source"]["default"] == "null"
    assert body["source"]["image_path"] == ""
    assert body["consumers"] == {
        "preview": True,
        "inference": True,
        "recording": False,
        "recording_format": "csv",
        "recording_path": "",
    }
    assert {
        "source.default",
        "source.image_path",
        "source.image_fps",
        "consumers.preview",
        "consumers.inference",
        "consumers.recording",
        "consumers.recording_format",
        "consumers.recording_path",
    }.issubset(field_paths)


def test_app_uses_csv_control_frame_recorder_by_default(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")

    assert isinstance(app.state.runtime.recorder, ControlFrameCsvRecorder)
    assert app.state.runtime.recorder.path == tmp_path / "data" / "recordings" / "control_frames.csv"


def test_app_parquet_recorder_requires_pyarrow_dependency(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.consumers.recording_format = "parquet"

    if ControlFrameParquetRecorder.available():
        app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml", config=cfg)
        assert isinstance(app.state.runtime.recorder, ControlFrameParquetRecorder)
        assert app.state.runtime.recorder.path == tmp_path / "data" / "recordings" / "control_frames.parquet"
    else:
        with pytest.raises(RuntimeError, match="pyarrow is required"):
            create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml", config=cfg)


def test_runtime_stop_endpoint_is_idempotent(tmp_path) -> None:
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=RuntimeConfig(),
    )
    client = TestClient(app)
    _activate(client)

    response = client.post("/api/runtime/stop")

    assert response.status_code == 200
    assert response.json()["running"] is False


def test_runtime_stop_clears_deepstream_pipeline_object(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream"
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    stopped: list[str] = []
    app.state.runtime.running = True
    app.state.runtime.pipeline = type(
        "Pipeline",
        (),
        {
            "stop": lambda self: stopped.append("stopped"),
            "status": lambda self: {"running": False, "detection_source": {"selected": "deepstream"}},
        },
    )()

    response = client.post("/api/runtime/stop")

    assert response.status_code == 200
    assert stopped == ["stopped"]
    assert response.json()["running"] is False
    assert app.state.runtime.pipeline is None
    assert app.state.runtime.running is False


def test_runtime_start_returns_400_when_capture_not_running(tmp_path) -> None:
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=RuntimeConfig(),
    )
    client = TestClient(app)
    _activate(client)

    response = client.post("/api/runtime/start")

    assert response.status_code == 400
    assert response.json()["detail"] == "采集未启动，无法运行推理链路。"
    assert app.state.runtime.running is False


def test_runtime_start_nvmm_latest_uses_runtime_owned_latest_branch(tmp_path, monkeypatch) -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "nvmm_latest"
    cfg.inference.enabled = False
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=cfg,
    )
    client = TestClient(app)
    _activate(client)
    monkeypatch.setattr(
        "novasight.api.routes_runtime.build_deepstream_detection_source",
        lambda _request: pytest.fail("nvmm_latest must not build Full DeepStream backend"),
    )
    app.state.capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
        wait_preview_frame=lambda *, after_frame_id=None, timeout_s=0.0: None,
    )

    response = client.post("/api/runtime/start")
    stop_response = client.post("/api/runtime/stop")

    assert response.status_code == 200
    assert response.json()["detection_source"] == {}
    assert stop_response.status_code == 200
    assert app.state.runtime.running is False


def test_runtime_local_trigger_endpoint_is_removed(tmp_path) -> None:
    app = create_app(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "missing.yaml",
        config=RuntimeConfig(),
    )
    client = TestClient(app)
    _activate(client)

    response = client.post(
        "/api/runtime/local-trigger",
        json={"active": True, "bindings": ["MouseLeft"]},
    )

    assert response.status_code == 404
    assert not hasattr(app.state.runtime, "update_local_trigger")


def test_runtime_calibration_fingerprint_api_updates_runtime_status(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)

    rejected = client.post("/api/runtime/calibration/fingerprint", json={"fingerprint": ""})
    accepted = client.post(
        "/api/runtime/calibration/fingerprint",
        json={
            "fingerprint": "unverified-default",
            "source": "game",
        },
    )
    state = client.get("/api/runtime/state").json()

    assert rejected.status_code == 400
    assert rejected.json()["detail"] == "sensitivity fingerprint must be non-empty"
    assert accepted.status_code == 200
    assert accepted.json()["state"] == "matched"
    assert accepted.json()["source"] == "game"
    assert state["vision"]["calibration"]["state"] == "matched"
    assert state["vision"]["calibration"]["observed_fingerprint"] == "unverified-default"


def test_license_api_stores_fingerprint_without_returning_plaintext(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)

    saved = client.post("/api/license/activate", json={"key": TEST_MAX_LICENSE_KEY}).json()
    status = client.get("/api/license").json()

    assert saved["configured"] is True
    assert saved["valid"] is True
    assert saved["tier"] == "test_max"
    assert "hardware_control" in saved["features"]
    assert status["fingerprint"] == saved["fingerprint"]
    assert TEST_MAX_LICENSE_KEY not in str(saved)
    assert (tmp_path / "data" / "license.json").exists()

    cleared = client.delete("/api/license").json()

    assert cleared["configured"] is False
