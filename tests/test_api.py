"""End-to-end tests for the model/project/version/artifact API surface.

Focuses on the routes' HTTP contract: status codes, response shape, and
side-effects that callers depend on (e.g. inference runtime reload on
publish). Validation of the underlying registry path rules is owned by
test_model_registry.py.
"""
from pathlib import Path

from fastapi.testclient import TestClient

import novasight.main as main_module
from novasight.api import create_app
from novasight.license import TEST_MAX_LICENSE_KEY


def _client(tmp_path: Path) -> TestClient:
    client = TestClient(
        create_app(
            data_dir=tmp_path / "data",
            config_path=tmp_path / "missing.yaml",
        )
    )
    response = client.post("/api/license/activate", json={"key": TEST_MAX_LICENSE_KEY})
    assert response.status_code == 200
    return client


def _project(
    client: TestClient,
    *,
    name: str = "demo",
    description: str = "Demo model",
) -> dict:
    response = client.post(
        "/api/models/projects",
        json={"name": name, "description": description},
    )
    assert response.status_code == 200
    return response.json()


def _version(client: TestClient, project_id: int, version: str = "v1") -> dict:
    response = client.post(
        f"/api/models/projects/{project_id}/versions",
        json={
            "version": version,
            "source_kind": "onnx",
            "source_path": f"/tmp/{version}.onnx",
            "classes": ["target"],
            "input_shape": "1x3x640x640",
        },
    )
    assert response.status_code == 200
    return response.json()


def _artifact(
    client: TestClient,
    version_id: int,
    *,
    path: str = "model.onnx",
    kind: str = "onnx",
    status: str = "ready",
) -> dict:
    response = client.post(
        f"/api/models/versions/{version_id}/artifacts",
        json={
            "kind": kind,
            "path": path,
            "checksum": f"sha256:{version_id}:{status}",
            "status": status,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_health_and_runtime_state(tmp_path: Path) -> None:
    client = _client(tmp_path)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    state = client.get("/api/runtime/state")
    body = state.json()
    assert state.status_code == 200
    assert body["executor"]["selected"] == "dry_run"
    assert body["active_model"] is None
    assert body["inference"]["selected"] == "unavailable"
    assert body["inference"]["available"] is False
    assert body["inference"]["loaded"] is False
    assert "TensorRT unavailable" in body["inference"]["reason"]


def test_model_publish_and_rollback(tmp_path: Path) -> None:
    client = _client(tmp_path)

    project = _project(client)
    v1 = _version(client, project["id"], version="v1")
    artifact1 = _artifact(client, v1["id"], path="v1.onnx")
    v2 = _version(client, project["id"], version="v2")
    artifact2 = _artifact(client, v2["id"], path="v2.onnx")

    assert client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": artifact1["id"]},
    ).status_code == 200
    published = client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": artifact2["id"]},
    )
    assert published.json()["previous_artifact_id"] == artifact1["id"]

    rolled_back = client.post(f"/api/models/projects/{project['id']}/rollback")
    assert rolled_back.status_code == 200
    assert rolled_back.json()["artifact_id"] == artifact1["id"]
    assert rolled_back.json()["previous_artifact_id"] == artifact2["id"]


def test_publish_engine_loads_inference_runtime(tmp_path: Path) -> None:
    client = _client(tmp_path)
    loaded: dict = {}

    class RecordingInference:
        def status(self) -> dict:
            return {"selected": "recording", "available": True, "loaded": False}

        def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
            loaded["artifact_path"] = artifact_path
            loaded["classes"] = classes
            loaded["input_shape"] = input_shape

    client.app.state.inference = RecordingInference()
    project = _project(client)
    version = _version(client, project["id"])
    artifact = _artifact(
        client, version["id"], path="model.engine", kind="engine"
    )

    response = client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": artifact["id"]},
    )

    assert response.status_code == 200
    assert loaded == {
        "artifact_path": tmp_path / "data" / "models" / project["name"] / version["version"] / "model.engine",
        "classes": ["target"],
        "input_shape": "1x3x640x640",
    }


def test_publish_non_engine_disables_previous_inference_runtime(tmp_path: Path) -> None:
    client = _client(tmp_path)
    events: list[tuple] = []

    class RecordingInference:
        def status(self) -> dict:
            return {"selected": "recording", "available": True, "loaded": True}

        def load(self, artifact_path: Path, classes: list[str], input_shape: str) -> None:
            events.append(("load", artifact_path.name, tuple(classes), input_shape))

        def disable(self, reason: str) -> None:
            events.append(("disable", reason))

    client.app.state.inference = RecordingInference()
    project = _project(client)
    v1 = _version(client, project["id"], version="v1")
    engine = _artifact(client, v1["id"], path="model.engine", kind="engine")
    v2 = _version(client, project["id"], version="v2")
    onnx = _artifact(client, v2["id"], path="model.onnx", kind="onnx")

    assert client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": engine["id"]},
    ).status_code == 200
    assert client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": onnx["id"]},
    ).status_code == 200

    assert events == [
        ("load", "model.engine", ("target",), "1x3x640x640"),
        ("disable", "published artifact is not TensorRT engine: onnx"),
    ]


def test_runtime_state_reports_latest_published_deployment(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first_project = _project(client, name="demo-a")
    first_version = _version(client, first_project["id"])
    first_artifact = _artifact(client, first_version["id"], path="demo-a.onnx")
    second_project = _project(client, name="demo-b")
    second_version = _version(client, second_project["id"])
    second_artifact = _artifact(client, second_version["id"], path="demo-b.onnx")

    assert client.post(
        f"/api/models/projects/{first_project['id']}/publish",
        json={"artifact_id": first_artifact["id"]},
    ).status_code == 200
    assert client.post(
        f"/api/models/projects/{second_project['id']}/publish",
        json={"artifact_id": second_artifact["id"]},
    ).status_code == 200

    state = client.get("/api/runtime/state").json()
    assert state["active_model"]["project"]["id"] == second_project["id"]
    assert state["active_model"]["artifact"]["id"] == second_artifact["id"]

    assert client.post(
        f"/api/models/projects/{first_project['id']}/publish",
        json={"artifact_id": first_artifact["id"]},
    ).status_code == 200

    updated = client.get("/api/runtime/state").json()["active_model"]
    assert updated["project"]["id"] == first_project["id"]
    assert updated["artifact"]["id"] == first_artifact["id"]


def test_prepare_yolov8n_example_downloads_and_registers_model(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path)
    downloads: list[tuple[str, Path]] = []

    def fake_download(url: str, path: Path) -> None:
        downloads.append((url, path))
        path.write_bytes(b"fake-yolov8n")

    monkeypatch.setattr(
        "novasight.api.routes_models._download_file",
        fake_download,
    )

    response = client.post("/api/models/examples/yolov8n/prepare")

    assert response.status_code == 200
    body = response.json()
    assert body["project"]["name"] == "yolov8n"
    assert body["version"]["version"] == "v8n"
    assert body["artifact"]["kind"] == "pt"
    assert body["artifact"]["status"] == "ready"
    assert body["downloaded"] is True
    assert downloads == [
        (
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            tmp_path / "data" / "models" / "yolov8n" / "v8n" / "yolov8n.pt",
        )
    ]

    again = client.post("/api/models/examples/yolov8n/prepare").json()

    assert again["downloaded"] is False
    assert client.get("/api/models/projects").json()[0]["name"] == "yolov8n"


def test_upload_model_file_registers_ready_artifact(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/models/upload",
        data={
            "project_name": "custom_model",
            "version": "v1",
            "description": "自定义模型",
            "classes": "target,ignore",
            "input_shape": "1x3x640x640",
        },
        files={"file": ("custom.onnx", b"onnx-bytes", "application/octet-stream")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["project"]["name"] == "custom_model"
    assert body["version"]["source_kind"] == "onnx"
    assert body["version"]["classes"] == ["target", "ignore"]
    assert body["artifact"]["kind"] == "onnx"
    assert body["artifact"]["path"] == "custom.onnx"
    assert body["artifact"]["status"] == "ready"
    assert (
        tmp_path
        / "data"
        / "models"
        / "custom_model"
        / "v1"
        / "custom.onnx"
    ).read_bytes() == b"onnx-bytes"


def test_plugin_and_executor_endpoints(tmp_path: Path) -> None:
    client = _client(tmp_path)

    plugins = client.get("/api/plugins").json()
    assert "control.center_target" in {item["plugin_id"] for item in plugins}
    assert all(item["enabled"] is True for item in plugins)

    executors = client.get("/api/executors").json()
    assert executors["selected"] == "dry_run"
    assert executors["executors"]["silent"]["available"] is True
    assert executors["executors"]["console"]["available"] is True
    assert executors["executors"]["dry_run"]["available"] is True


def test_conversion_job_lifecycle(tmp_path: Path) -> None:
    client = _client(tmp_path)
    project = _project(client)
    version = _version(client, project["id"])

    created = client.post(
        f"/api/models/versions/{version['id']}/jobs",
        json={"target_kind": "engine", "command": ["trtexec", "--onnx=demo.onnx"]},
    )
    assert created.status_code == 200
    job = created.json()
    assert job["status"] == "pending"

    assert client.get("/api/models/jobs").json() == [job]
    assert client.get(f"/api/models/jobs?version_id={version['id']}").json() == [job]
    assert client.get(f"/api/models/jobs/{job['id']}").json() == job

    finished = client.post(
        f"/api/models/jobs/{job['id']}/finish",
        json={"status": "succeeded", "log": "ok"},
    )
    assert finished.status_code == 200
    assert finished.json()["ok"] is True

    reloaded = client.get(f"/api/models/jobs/{job['id']}").json()
    assert reloaded["status"] == "succeeded"
    assert reloaded["log"] == "ok"


def test_unknown_ids_return_404_and_invalid_state_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    project = _project(client)
    version = _version(client, project["id"])
    pending_artifact = _artifact(client, version["id"], status="pending")

    assert client.post("/api/models/versions/999/artifacts", json={
        "kind": "onnx", "path": "/tmp/missing.onnx", "checksum": "sha256:missing", "status": "ready",
    }).status_code == 404
    assert client.get("/api/models/jobs/999").status_code == 404
    assert client.get("/api/models/projects/999/versions").status_code == 404
    assert client.get("/api/models/versions/999/artifacts").status_code == 404
    assert client.get("/api/models/jobs?version_id=999").status_code == 404
    assert client.post(
        "/api/models/jobs/999/finish",
        json={"status": "succeeded", "log": "missing"},
    ).status_code == 404
    assert client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": 999},
    ).status_code == 404
    assert client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": pending_artifact["id"]},
    ).status_code == 400
    assert client.post(
        f"/api/models/versions/{version['id']}/artifacts",
        json={
            "kind": "bad", "path": "/tmp/demo.onnx",
            "checksum": "sha256:bad", "status": "ready",
        },
    ).status_code == 400
    assert client.post("/api/models/projects/999/rollback").status_code == 404
    no_deployment = client.post(f"/api/models/projects/{project['id']}/rollback")
    assert no_deployment.status_code == 400
    assert "no deployment" in no_deployment.json()["detail"]


def test_duplicate_project_and_version_return_controlled_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    project = _project(client)
    _version(client, project["id"])

    assert client.post(
        "/api/models/projects", json={"name": "demo", "description": "Duplicate"}
    ).json()["detail"] == "project already exists: demo"

    dup_version = client.post(
        f"/api/models/projects/{project['id']}/versions",
        json={
            "version": "v1", "source_kind": "onnx",
            "source_path": "/tmp/duplicate.onnx",
            "classes": ["target"], "input_shape": "1x3x640x640",
        },
    )
    assert dup_version.status_code == 400
    assert "already exists" in dup_version.json()["detail"]


def test_malformed_payloads_return_422(tmp_path: Path) -> None:
    client = _client(tmp_path)
    project = _project(client)

    bad_classes = client.post(
        f"/api/models/projects/{project['id']}/versions",
        json={
            "version": "v1", "source_kind": "onnx",
            "source_path": "/tmp/demo.onnx",
            "classes": "abc", "input_shape": "1x3x640x640",
        },
    )
    assert bad_classes.status_code == 422

    bad_project = client.post(
        "/api/models/projects",
        json={"name": None, "description": "Demo model"},
    )
    assert bad_project.status_code == 422


def test_cli_overrides_are_visible_to_uvicorn_and_app_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_run(app, host: str, port: int) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    result = main_module.main(
        [
            "--host", "127.0.0.1",
            "--port", "6001",
            "--data-dir", str(tmp_path / "data"),
            "--config", str(tmp_path / "missing.yaml"),
        ]
    )

    assert result == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 6001
    assert captured["app"].state.config.web.host == "127.0.0.1"
    assert captured["app"].state.config.web.port == 6001
