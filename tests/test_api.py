from pathlib import Path

from fastapi.testclient import TestClient

import novasight.main as main_module
from novasight.api import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            data_dir=tmp_path / "data",
            config_path=tmp_path / "missing.yaml",
        )
    )


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
    status: str = "ready",
) -> dict:
    response = client.post(
        f"/api/models/versions/{version_id}/artifacts",
        json={
            "kind": "onnx",
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
    assert state.status_code == 200
    body = state.json()
    assert body["executor"]["selected"] == "dry_run"
    assert body["active_model"] is None
    assert body["inference"]["selected"] == "unavailable"
    assert body["inference"]["available"] is False
    assert body["inference"]["loaded"] is False
    assert "TensorRT unavailable" in body["inference"]["reason"]


def test_model_project_version_artifact_publish_flow(tmp_path: Path) -> None:
    client = _client(tmp_path)

    project = _project(client)
    version = _version(client, project["id"])
    artifact = _artifact(client, version["id"])

    published = client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": artifact["id"]},
    )

    assert published.status_code == 200
    assert published.json()["artifact_id"] == artifact["id"]
    assert artifact["path"] == "model.onnx"


def test_artifact_create_rejects_absolute_path_outside_version_assets(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    project = _project(client)
    version = _version(client, project["id"])

    response = client.post(
        f"/api/models/versions/{version['id']}/artifacts",
        json={
            "kind": "onnx",
            "path": "/tmp/demo.onnx",
            "checksum": "sha256:absolute",
            "status": "ready",
        },
    )

    assert response.status_code == 400
    assert "artifact path" in response.json()["detail"]


def test_artifact_create_rejects_absolute_path_inside_version_assets(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    project = _project(client)
    version = _version(client, project["id"])
    inside_path = tmp_path / "data" / "models" / "demo" / "v1" / "model.onnx"

    response = client.post(
        f"/api/models/versions/{version['id']}/artifacts",
        json={
            "kind": "onnx",
            "path": str(inside_path),
            "checksum": "sha256:absolute-inside",
            "status": "ready",
        },
    )

    assert response.status_code == 400
    assert "relative" in response.json()["detail"]


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

    state = client.get("/api/runtime/state")
    assert state.status_code == 200
    active_model = state.json()["active_model"]
    assert active_model["project"]["id"] == second_project["id"]
    assert active_model["artifact"]["id"] == second_artifact["id"]

    assert client.post(
        f"/api/models/projects/{first_project['id']}/publish",
        json={"artifact_id": first_artifact["id"]},
    ).status_code == 200

    updated_state = client.get("/api/runtime/state")
    assert updated_state.status_code == 200
    updated_active_model = updated_state.json()["active_model"]
    assert updated_active_model["project"]["id"] == first_project["id"]
    assert updated_active_model["artifact"]["id"] == first_artifact["id"]


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


def test_conversion_job_create_list_get_finish_flow(tmp_path: Path) -> None:
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

    listed = client.get("/api/models/jobs")
    assert listed.status_code == 200
    assert listed.json() == [job]

    scoped = client.get(f"/api/models/jobs?version_id={version['id']}")
    assert scoped.status_code == 200
    assert scoped.json() == [job]

    loaded = client.get(f"/api/models/jobs/{job['id']}")
    assert loaded.status_code == 200
    assert loaded.json() == job

    finished = client.post(
        f"/api/models/jobs/{job['id']}/finish",
        json={"status": "succeeded", "log": "ok"},
    )
    assert finished.status_code == 200
    assert finished.json()["ok"] is True

    reloaded = client.get(f"/api/models/jobs/{job['id']}").json()
    assert reloaded["status"] == "succeeded"
    assert reloaded["log"] == "ok"


def test_publish_and_rollback_restores_previous_artifact(tmp_path: Path) -> None:
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
    assert published.status_code == 200
    assert published.json()["previous_artifact_id"] == artifact1["id"]

    rolled_back = client.post(f"/api/models/projects/{project['id']}/rollback")
    assert rolled_back.status_code == 200
    assert rolled_back.json()["artifact_id"] == artifact1["id"]
    assert rolled_back.json()["previous_artifact_id"] == artifact2["id"]


def test_unknown_ids_return_404_and_invalid_state_returns_400(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    project = _project(client)
    version = _version(client, project["id"])
    pending_artifact = _artifact(client, version["id"], status="pending")

    unknown_version = client.post(
        "/api/models/versions/999/artifacts",
        json={
            "kind": "onnx",
            "path": "/tmp/missing.onnx",
            "checksum": "sha256:missing",
            "status": "ready",
        },
    )
    assert unknown_version.status_code == 404

    unknown_job = client.get("/api/models/jobs/999")
    assert unknown_job.status_code == 404

    unknown_versions_list = client.get("/api/models/projects/999/versions")
    assert unknown_versions_list.status_code == 404

    unknown_artifacts_list = client.get("/api/models/versions/999/artifacts")
    assert unknown_artifacts_list.status_code == 404

    unknown_scoped_jobs = client.get("/api/models/jobs?version_id=999")
    assert unknown_scoped_jobs.status_code == 404

    unknown_finish = client.post(
        "/api/models/jobs/999/finish",
        json={"status": "succeeded", "log": "missing"},
    )
    assert unknown_finish.status_code == 404

    unknown_artifact = client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": 999},
    )
    assert unknown_artifact.status_code == 404

    unknown_project_publish = client.post(
        "/api/models/projects/999/publish",
        json={"artifact_id": pending_artifact["id"]},
    )
    assert unknown_project_publish.status_code == 404

    not_ready = client.post(
        f"/api/models/projects/{project['id']}/publish",
        json={"artifact_id": pending_artifact["id"]},
    )
    assert not_ready.status_code == 400

    invalid_kind = client.post(
        f"/api/models/versions/{version['id']}/artifacts",
        json={
            "kind": "bad",
            "path": "/tmp/demo.onnx",
            "checksum": "sha256:bad",
            "status": "ready",
        },
    )
    assert invalid_kind.status_code == 400


def test_rollback_unknown_project_is_404_but_missing_deployment_is_400(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    project = _project(client)

    unknown_project = client.post("/api/models/projects/999/rollback")
    assert unknown_project.status_code == 404

    no_deployment = client.post(f"/api/models/projects/{project['id']}/rollback")
    assert no_deployment.status_code == 400
    assert "no deployment" in no_deployment.json()["detail"]


def test_duplicate_project_and_version_return_controlled_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    project = _project(client)
    _version(client, project["id"])

    duplicate_project = client.post(
        "/api/models/projects",
        json={"name": "demo", "description": "Duplicate"},
    )
    assert duplicate_project.status_code == 400
    assert "already exists" in duplicate_project.json()["detail"]

    duplicate_version = client.post(
        f"/api/models/projects/{project['id']}/versions",
        json={
            "version": "v1",
            "source_kind": "onnx",
            "source_path": "/tmp/duplicate.onnx",
            "classes": ["target"],
            "input_shape": "1x3x640x640",
        },
    )
    assert duplicate_version.status_code == 400
    assert "already exists" in duplicate_version.json()["detail"]


def test_malformed_payloads_return_422_without_string_coercion(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    project = _project(client)

    bad_classes = client.post(
        f"/api/models/projects/{project['id']}/versions",
        json={
            "version": "v1",
            "source_kind": "onnx",
            "source_path": "/tmp/demo.onnx",
            "classes": "abc",
            "input_shape": "1x3x640x640",
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
            "--host",
            "127.0.0.1",
            "--port",
            "6001",
            "--data-dir",
            str(tmp_path / "data"),
            "--config",
            str(tmp_path / "missing.yaml"),
        ]
    )

    assert result == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 6001
    assert captured["app"].state.config.web.host == "127.0.0.1"
    assert captured["app"].state.config.web.port == 6001
