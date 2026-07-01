from pathlib import Path

from fastapi.testclient import TestClient

from novasight.api import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(data_dir=tmp_path / "data"))


def _project(client: TestClient) -> dict:
    response = client.post(
        "/api/models/projects",
        json={"name": "demo", "description": "Demo model"},
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
    path: str = "/tmp/demo.onnx",
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
    artifact_path = Path(artifact["path"])
    assert artifact_path != Path("/tmp/demo.onnx")
    assert artifact_path.is_relative_to(tmp_path / "data" / "models" / "demo" / "v1")


def test_plugin_and_executor_endpoints(tmp_path: Path) -> None:
    client = _client(tmp_path)

    plugins = client.get("/api/plugins").json()
    assert "control.center_target" in {item["plugin_id"] for item in plugins}

    executors = client.get("/api/executors").json()
    assert executors["selected"] == "dry_run"
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
    artifact1 = _artifact(client, v1["id"], path="/tmp/v1.onnx")
    v2 = _version(client, project["id"], version="v2")
    artifact2 = _artifact(client, v2["id"], path="/tmp/v2.onnx")

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
