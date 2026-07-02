from __future__ import annotations

from fastapi.testclient import TestClient

from novasight.api import create_app
from novasight.config import RuntimeConfig
from novasight.license import TEST_MAX_LICENSE_KEY


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
    body["limits"]["max_frame_queue"] = 1

    response = client.put("/api/config", json=body)

    assert response.status_code == 200
    assert response.json()["config"]["capture"]["device"] == "/dev/video1"
    assert app.state.runtime.config_store.status()["version"] == 1


def test_config_schema_matches_runtime_config_and_update_syncs_runtime_objects(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = TestClient(app)
    _activate(client)

    schema = client.get("/api/config/schema").json()
    body = schema["values"]
    body["control"]["output_mode"] = "silent"
    body["capture"]["device"] = "/dev/video7"

    response = client.put("/api/config", json=body)

    assert response.status_code == 200
    assert response.json()["config"]["capture"]["device"] == "/dev/video7"
    assert app.state.capture.config.device == "/dev/video7"
    assert app.state.runtime.config.capture.device == "/dev/video7"
    assert app.state.runtime.executors.selected == "silent"
    assert any(section["id"] == "hardware" for section in schema["sections"])


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
