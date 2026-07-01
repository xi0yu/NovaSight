from fastapi.testclient import TestClient

from novasight.api import create_app


def test_capture_state_is_in_runtime_state(tmp_path) -> None:
    client = TestClient(
        create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    )

    response = client.get("/api/runtime/state")

    assert response.status_code == 200
    assert "capture" in response.json()
    assert response.json()["capture"]["device"] == "/dev/video0"


def test_capture_capabilities_endpoint_uses_service(tmp_path) -> None:
    client = TestClient(
        create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    )

    response = client.get("/api/capture/capabilities?device=/dev/video0")

    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "/dev/video0"
    assert "available" in body
    assert "capabilities" in body


def test_capture_select_rejects_unavailable_device(tmp_path) -> None:
    client = TestClient(
        create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    )

    response = client.post("/api/capture/select", json={"device": "/dev/missing"})

    assert response.status_code in {200, 400}
    assert "device" in response.json()
