"""HTTP-level tests for the /api/capture routes.

Cover the request validation, the success path, and the failure modes that
matter for the React workbench: device must be present, capability listing
goes through the service, the service config is never mutated on failure.
"""
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from novasight.api import create_app
from novasight.api.routes_capture import _mjpeg_frames, stream
from novasight.capture.service import CaptureService
from novasight.capture.source import CapturedFrame
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
    source: object | None = None
    stop_calls: list[str] | None = None

    def configure(self, device: str, **kwargs) -> CaptureRuntimeState:
        self.configure_calls.append(device)
        return self.state

    def stop(self, reason: str | None = None) -> CaptureRuntimeState:
        if self.stop_calls is not None:
            self.stop_calls.append(reason or "")
        return self.state


class ReadFrameForbiddenCapture:
    empty_read_sleep_s = 0

    def __init__(self, frames: list[CapturedFrame | None]) -> None:
        self.frames = frames
        self.state = CaptureRuntimeState(available=True)
        self.preview_drops = 0
        self.preview_outputs = 0

    def wait_preview_frame(
        self,
        *,
        after_frame_id: int | None = None,
        timeout_s: float = 0.0,
    ) -> CapturedFrame | None:
        del after_frame_id, timeout_s
        if not self.frames:
            return None
        return self.frames.pop(0)

    def record_preview_output(self, frame: CapturedFrame, *, target_fps: int) -> None:
        del frame, target_fps
        self.preview_outputs += 1

    def record_preview_drop(self, *, target_fps: int) -> None:
        del target_fps
        self.preview_drops += 1


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
            return None
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


def _preview_frame(frame_id: int = 1) -> CapturedFrame:
    from PIL import Image

    return CapturedFrame(
        frame_id=frame_id,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=1_000_000_000 + frame_id,
        capture_wait_ms=1.0,
        image=Image.new("RGB", (2, 2), (0, 0, 0)),
    )


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


def test_capture_select_failure_stops_previous_capture(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT if device == "/dev/video0" else None,
        source_factory=lambda profile: CachedPreviewSource(),
    )
    service.configure("/dev/video0")
    app.state.capture = service
    client = _client(app)

    response = client.post("/api/capture/select", json={"device": "/dev/missing"})

    assert response.status_code == 400
    state = client.get("/api/capture/state").json()
    assert state["available"] is False
    assert state["device"] == "/dev/missing"
    assert state["last_error"] is not None


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


def test_capture_stream_returns_503_when_capture_session_not_running(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    service = FakeCaptureService(
        state=CaptureRuntimeState(available=False, device="/dev/video0"),
        config=RuntimeConfig().capture,
        configure_calls=[],
    )
    app.state.capture = service
    client = _client(app)

    response = client.get("/api/capture/stream.mjpg")

    assert response.status_code == 503
    assert response.json()["message"] == "采集未启动，无法打开预览。"
    assert service.configure_calls == []


def test_capture_stream_returns_503_when_capture_state_unavailable() -> None:
    capture = FakeCaptureService(
        state=CaptureRuntimeState(available=False, device="/dev/video0"),
        config=RuntimeConfig().capture,
        configure_calls=[],
        source=object(),
    )
    capture.session = SimpleNamespace(running=True)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                capture=capture,
                config=RuntimeConfig(),
                runtime=None,
            )
        )
    )

    response = stream(request)

    assert response.status_code == 503
    assert json.loads(response.body)["message"] == "采集未启动，无法打开预览。"
    assert capture.configure_calls == []


def test_capture_select_starts_session_for_preview(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: StreamingSource(),
    )
    app.state.capture = service
    client = _client(app)

    try:
        select_response = client.post("/api/capture/select", json={"device": "/dev/video0"})
        source_started = service.source is not None
        state_available = service.state.available
        assert service.wait_preview_frame(after_frame_id=0, timeout_s=0.2) is not None
        payload = next(_mjpeg_frames(service, preview_fps=30, max_frames=1))
    finally:
        service.stop("test complete")

    assert select_response.status_code == 200
    assert source_started is True
    assert state_available is True
    assert b"Content-Type: image/jpeg" in payload
    assert b"\xff\xd8" in payload


def test_capture_stream_uses_preview_cache(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    cfg = RuntimeConfig()
    source = CachedPreviewSource()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: source,
    )
    service.configure("/dev/video0")
    assert service.wait_preview_frame(after_frame_id=0, timeout_s=0.2) is not None
    app.state.capture = service
    app.state.config.limits.stream_fps = 30

    try:
        payload = next(_mjpeg_frames(service, preview_fps=30, max_frames=1))
    finally:
        service.stop("test complete")

    assert b"Content-Type: image/jpeg" in payload
    assert service.state.preview_output_frames >= 1


def test_mjpeg_frames_waits_for_preview_without_direct_read() -> None:
    capture = ReadFrameForbiddenCapture([_preview_frame()])

    payload = next(_mjpeg_frames(capture, preview_fps=30, max_frames=1))

    assert b"Content-Type: image/jpeg" in payload
    assert capture.preview_outputs == 1


def test_mjpeg_frames_passes_roi_size_to_preview_renderer(monkeypatch) -> None:
    calls: list[int] = []

    def fake_render_preview_frame(frame, *, runtime=None, roi_size=640):
        del runtime
        calls.append(roi_size)
        return frame.image

    monkeypatch.setattr(
        "novasight.api.routes_capture.render_preview_frame",
        fake_render_preview_frame,
    )
    capture = ReadFrameForbiddenCapture([_preview_frame()])

    payload = next(_mjpeg_frames(capture, preview_fps=30, roi_size=256, max_frames=1))

    assert b"Content-Type: image/jpeg" in payload
    assert calls == [256]


def test_mjpeg_frames_timeout_records_preview_drop_only() -> None:
    capture = ReadFrameForbiddenCapture([None])

    assert list(_mjpeg_frames(capture, preview_fps=30, max_attempts=1)) == []

    assert capture.preview_drops == 1
    assert capture.preview_outputs == 0


def test_mjpeg_frames_max_frames_counts_emitted_frames() -> None:
    capture = ReadFrameForbiddenCapture([None, _preview_frame()])

    payloads = list(_mjpeg_frames(capture, preview_fps=30, max_frames=1, max_attempts=2))

    assert len(payloads) == 1
    assert b"Content-Type: image/jpeg" in payloads[0]
    assert capture.preview_drops == 1
    assert capture.preview_outputs == 1


def test_capture_stop_delegates_to_service(tmp_path) -> None:
    app = create_app(data_dir=tmp_path / "data", config_path=tmp_path / "missing.yaml")
    service = FakeCaptureService(
        state=CaptureRuntimeState(available=False, device="/dev/video0"),
        config=RuntimeConfig().capture,
        configure_calls=[],
        stop_calls=[],
    )
    app.state.capture = service
    client = _client(app)

    response = client.post("/api/capture/stop")

    assert response.status_code == 200
    assert service.stop_calls == ["capture stopped by user"]
