"""HTTP-level tests for the /api/capture routes.

Cover the request validation, the success path, and the failure modes that
matter for the React workbench: device must be present, capability listing
goes through the service, the service config is never mutated on failure.
"""
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from novasight.api import create_app
from novasight.api.routes_capture import _mjpeg_frames
from novasight.capture.service import CaptureService
from novasight.capture.state import CaptureRuntimeState
from novasight.config import RuntimeConfig
from novasight.license import TEST_MAX_LICENSE_KEY


CAPS_TEXT = "[0]: 'MJPG' (Motion-JPEG)\n    Size: Discrete 1280x720\n        Interval: Discrete 0.017s (60.000 fps)\n"


def _client(app) -> TestClient:
    client = TestClient(app)
    response = client.post("/api/license/activate", json={"key": TEST_MAX_LICENSE_KEY})
    assert response.status_code == 200
    return client


@dataclass
class FakeCaptureService:
    state: CaptureRuntimeState
    config: object
    configure_calls: list[str]

    def configure(self, device: str, **kwargs) -> CaptureRuntimeState:
        self.configure_calls.append(device)
        return self.state


class FakeSource:
    backend_label = "gst:test"

    def close(self) -> None:
        pass


class StreamingSource:
    backend_label = "gst:stream"

    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def read(self):
        from PIL import Image

        from novasight.capture.source import CapturedFrame

        self.count += 1
        if self.count > 1:
            raise RuntimeError("stream complete")
        return CapturedFrame(
            frame_id=self.count,
            width=2,
            height=2,
            pixel_format="BGR",
            ts_ns=1_000_000_000,
            capture_wait_ms=1.0,
            image=Image.new("RGB", (2, 2), (0, 0, 0)),
        )

    def close(self) -> None:
        self.closed = True


class CachedPreviewSource:
    backend_label = "gst:cached-preview"

    def __init__(self) -> None:
        self.count = 0

    def read(self):
        from PIL import Image

        from novasight.capture.source import CapturedFrame

        self.count += 1
        return CapturedFrame(
            frame_id=self.count,
            width=2,
            height=2,
            pixel_format="BGR",
            ts_ns=1_000_000_000 + self.count,
            capture_wait_ms=1.0,
            image=Image.new("RGB", (2, 2), (0, 0, 0)),
        )

    def close(self) -> None:
        pass


def test_capture_state_is_in_runtime_state(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    client = _client(app)

    response = client.get("/api/runtime/state").json()

    assert response["capture"]["device"] == "/dev/video0"


def test_capture_capabilities_endpoint_uses_service(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    app.state.capture = CaptureService(
        config=RuntimeConfig().capture,
        capability_runner=lambda device: CAPS_TEXT if device == "/dev/fake" else None,
    )
    client = _client(app)
    body = client.get("/api/capture/capabilities?device=/dev/fake").json()

    assert body["device"] == "/dev/fake"
    assert body["available"] is True
    assert body["capabilities"][0] == {
        "pixel_format": "MJPG",
        "width": 1280,
        "height": 720,
        "fps_list": [60.0],
    }


@pytest.mark.parametrize("payload", [{}, {"device": "  "}])
def test_capture_select_requires_device_without_configuring(tmp_path, payload) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    service = FakeCaptureService(
        state=CaptureRuntimeState(device="/dev/video0"),
        config=RuntimeConfig().capture,
        configure_calls=[],
    )
    app.state.capture = service
    client = _client(app)

    assert client.post("/api/capture/select", json=payload).status_code == 422
    assert service.configure_calls == []


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
    client = _client(app)

    body = client.post("/api/capture/select", json={"device": "/dev/missing"}).json()

    assert body["device"] == "/dev/missing"
    assert body["available"] is False
    assert body["last_error"] == "device missing"


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"preference": "manual", "pixel_format": "MJPG",
          "width": 1920, "height": 1080, "fps": 144}, "device missing"),
        ({"preference": "invalid", "pixel_format": "MJPG",
          "width": 1280, "height": 720, "fps": 60}, "invalid"),
    ],
)
def test_capture_select_failure_preserves_service_config(
    tmp_path, extra: dict, match: str
) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    cfg.capture.preference = "auto_balanced"
    cfg.capture.pixel_format = "NV12"
    cfg.capture.width = 640
    cfg.capture.height = 480
    cfg.capture.fps = 30
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: FakeSource(),
    )
    app.state.capture = service
    client = _client(app)

    response = client.post(
        "/api/capture/select", json={"device": "/dev/video0", **extra}
    )

    assert response.status_code == 400
    assert service.config.preference == "auto_balanced"
    assert service.config.pixel_format == "NV12"
    assert service.config.width == 640
    assert service.config.height == 480
    assert service.config.fps == 30
    assert match in response.text or response.json()["last_error"] is not None


def test_capture_select_failure_keeps_previous_healthy_state(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT if device == "/dev/video0" else None,
        source_factory=lambda profile: FakeSource(),
    )
    service.configure("/dev/video0")
    app.state.capture = service
    client = _client(app)

    response = client.post("/api/capture/select", json={"device": "/dev/missing"})

    assert response.status_code == 400
    state = client.get("/api/capture/state").json()
    assert state["available"] is True
    assert state["device"] == "/dev/video0"
    assert state["last_error"] is None


def test_capture_select_applies_preference_to_service_config(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: FakeSource(),
    )
    app.state.capture = service
    client = _client(app)

    response = client.post(
        "/api/capture/select",
        json={"device": "/dev/video0", "preference": "auto_low_latency"},
    )

    assert response.status_code == 200
    assert service.config.preference == "auto_low_latency"


def test_capture_stream_returns_mjpeg_from_configured_source(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: StreamingSource(),
    )
    app.state.capture = service
    client = _client(app)

    response = client.get("/api/capture/stream.mjpg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"Content-Type: image/jpeg" in response.content
    assert b"\xff\xd8" in response.content
    assert service.state.available is False
    assert "stream complete" in str(service.state.last_error)


def test_capture_stream_uses_preview_cache_without_reading_source_again(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    source = CachedPreviewSource()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: source,
    )
    service.configure("/dev/video0")
    service.read_frame()
    app.state.capture = service
    app.state.config.limits.stream_fps = 30

    payload = next(_mjpeg_frames(service, preview_fps=30, max_frames=1))

    assert b"Content-Type: image/jpeg" in payload
    assert source.count == 1
    assert service.state.preview_output_frames >= 1
