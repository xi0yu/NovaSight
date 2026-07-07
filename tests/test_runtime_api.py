from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from novasight.api import create_app
from novasight.config import RuntimeConfig
from novasight.inference.jetson import JetsonGpuResourcePreprocessor
from novasight.license import TEST_MAX_LICENSE_KEY
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


def test_device_capabilities_describe_jetson_runtime_boundary(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
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
    assert body["endpoints"]["model_scan"] == "/api/models/scan"
    assert body["features"]["model_management"] is True
    assert body["features"]["remote_runtime_control"] is True
    assert body["features"]["deepstream_pipeline_generation"] is True
    assert body["features"]["deepstream_runtime_backend"] is False


def test_model_scan_api_reports_manifest_configuration_status(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)
    models_dir = app.state.models.data_dir
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
    assert statuses["need_confirm/default/model.engine"]["status"] == "need_confirm"
    assert statuses["need_confirm/default/model.engine"]["reason"] == "model manifest is missing"
    assert body["artifact_status_counts"]["ready"] == 1
    assert body["artifact_status_counts"]["need_confirm"] == 1


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
            "class_count": 4,
            "confidence_threshold": 0.35,
            "nms_iou_threshold": 0.5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["manifest_path"] == "combat_model/default/model.manifest.json"
    assert body["deepstream_config_path"] == "combat_model/default/deepstream.ini"
    manifest = read_manifest(model_dir / "model.manifest.json")
    assert manifest.model_id == "combat_model"
    assert manifest.display_name == "Combat Model"
    assert manifest.input.shape == [1, 3, 256, 256]
    assert manifest.output.shape == [1, 8, 1344]
    assert manifest.output.class_count == 4
    assert manifest.postprocess.confidence_threshold == 0.35
    deepstream_ini = (model_dir / "deepstream.ini").read_text(encoding="utf-8")
    assert f"model-engine-file={engine_path.resolve()}" in deepstream_ini
    assert "network-type=100" in deepstream_ini
    assert "output-tensor-meta=1" in deepstream_ini
    assert "output-blob-names=output0" in deepstream_ini


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
    pipeline = body["pipeline"]
    assert "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12,width=256,height=256" in pipeline
    assert "nvstreammux name=mux batch-size=1 width=256 height=256 live-source=1" in pipeline
    assert f"nvinfer name=primary-infer config-file-path={(model_dir / 'deepstream.ini').resolve()}" in pipeline
    assert "videoconvert" not in pipeline


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
