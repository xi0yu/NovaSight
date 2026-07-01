from dataclasses import dataclass

from fastapi.testclient import TestClient

from novasight.api import create_app
from novasight.capture.service import CaptureService
from novasight.capture.state import CaptureRuntimeState
from novasight.config import RuntimeConfig


CAPS_TEXT = "[0]: 'MJPG' (Motion-JPEG)\n    Size: Discrete 1280x720\n        Interval: Discrete 0.017s (60.000 fps)\n"


@dataclass
class FakeCaptureService:
    state: CaptureRuntimeState
    config: object
    configure_calls: list[str]

    def configure(self, device: str) -> CaptureRuntimeState:
        self.configure_calls.append(device)
        return self.state


def test_capture_state_is_in_runtime_state(tmp_path) -> None:
    client = TestClient(
        create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    )

    response = client.get("/api/runtime/state")

    assert response.status_code == 200
    assert "capture" in response.json()
    assert response.json()["capture"]["device"] == "/dev/video0"


def test_capture_capabilities_endpoint_uses_service(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    app.state.capture = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT if device == "/dev/fake" else None,
    )
    client = TestClient(app)

    response = client.get("/api/capture/capabilities?device=/dev/fake")

    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "/dev/fake"
    assert body["available"] is True
    assert body["capabilities"][0]["pixel_format"] == "MJPG"


def test_capture_select_rejects_unavailable_device(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    app.state.capture = FakeCaptureService(
        state=CaptureRuntimeState(
            available=False,
            device="/dev/missing",
            last_error="device missing",
        ),
        config=RuntimeConfig().capture,
        configure_calls=[],
    )
    client = TestClient(app)

    response = client.post("/api/capture/select", json={"device": "/dev/missing"})

    assert response.status_code == 400
    body = response.json()
    assert body["device"] == "/dev/missing"
    assert body["available"] is False
    assert body["last_error"] == "device missing"


def test_capture_select_requires_device_without_configuring(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    service = FakeCaptureService(
        state=CaptureRuntimeState(device="/dev/video0"),
        config=RuntimeConfig().capture,
        configure_calls=[],
    )
    app.state.capture = service
    client = TestClient(app)

    response = client.post("/api/capture/select", json={})

    assert response.status_code == 422
    assert service.configure_calls == []


def test_capture_select_applies_preference_to_service_config(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    service = FakeCaptureService(
        state=CaptureRuntimeState(available=True, device="/dev/video0"),
        config=RuntimeConfig().capture,
        configure_calls=[],
    )
    app.state.capture = service
    client = TestClient(app)

    response = client.post(
        "/api/capture/select",
        json={"device": "/dev/video0", "preference": "manual"},
    )

    assert response.status_code == 200
    assert service.config.preference == "manual"
    assert service.configure_calls == ["/dev/video0"]
