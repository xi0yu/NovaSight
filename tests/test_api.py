from pathlib import Path

from fastapi.testclient import TestClient

from novasight.api import create_app


def test_health_and_runtime_state(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    state = client.get("/api/runtime/state")
    assert state.status_code == 200
    body = state.json()
    assert body["executor"]["selected"] == "dry_run"
    assert body["active_model"] is None


def test_model_project_version_artifact_publish_flow(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)

    project = client.post(
        "/api/models/projects",
        json={"name": "demo", "description": "Demo model"},
    ).json()
    version = client.post(
        f"/api/models/projects/{project['id']}/versions",
        json={
            "version": "v1",
            "source_kind": "onnx",
            "source_path": "/tmp/demo.onnx",
            "classes": ["target"],
            "input_shape": "1x3x640x640",
        },
    ).json()
    artifact = client.post(
        f"/api/models/versions/{version['id']}/artifacts",
        json={
            "kind": "onnx",
            "path": "/tmp/demo.onnx",
            "checksum": "sha256:demo",
            "status": "ready",
        },
    ).json()

    published = client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": artifact["id"]},
    )

    assert published.status_code == 200
    assert published.json()["artifact_id"] == artifact["id"]


def test_plugin_and_executor_endpoints(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path / "data")
    client = TestClient(app)

    plugins = client.get("/api/plugins").json()
    assert "control.center_target" in {item["plugin_id"] for item in plugins}

    executors = client.get("/api/executors").json()
    assert executors["selected"] == "dry_run"
    assert executors["executors"]["dry_run"]["available"] is True
