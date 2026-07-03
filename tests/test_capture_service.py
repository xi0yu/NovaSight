"""Tests for CaptureService orchestration around the capture session.

The service owns capability selection, configuration persistence, and session
lifecycle coordination. Frame reads stay inside CaptureSession.
"""
import threading
import time

import pytest

from novasight.capture.service import CaptureService
from novasight.config import RuntimeConfig


CAPS_TEXT = "[0]: 'MJPG' (Motion-JPEG)\n    Size: Discrete 1920x1080\n        Interval: Discrete 0.007s (144.000 fps)\n"
MULTI_CAPS_TEXT = """
[0]: 'MJPG' (Motion-JPEG)
    Size: Discrete 1920x1080
        Interval: Discrete 0.008s (120.000 fps)
        Interval: Discrete 0.017s (60.000 fps)
"""
_SERVICES: list[CaptureService] = []


class FakeSource:
    backend_label = "gst:test"

    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def read(self):
        from novasight.capture.source import CapturedFrame

        self.count += 1
        return CapturedFrame(
            frame_id=self.count, width=1920, height=1080, pixel_format="BGR",
            ts_ns=1_000_000_000 + self.count * 7_000_000, capture_wait_ms=6.5, image=None,
        )

    def close(self) -> None:
        self.closed = True


class ExplodingSource:
    backend_label = "gst:broken"

    def __init__(self) -> None:
        self.closed = False

    def read(self):
        raise RuntimeError("camera disconnected")

    def close(self) -> None:
        self.closed = True


class ControlledSource:
    backend_label = "gst:controlled"

    def __init__(self) -> None:
        self.closed = False
        self._frames = []
        self._condition = threading.Condition()

    def emit(self, frame_id: int) -> None:
        from novasight.capture.source import CapturedFrame

        with self._condition:
            self._frames.append(
                CapturedFrame(
                    frame_id=frame_id,
                    width=1920,
                    height=1080,
                    pixel_format="BGR",
                    ts_ns=1_000_000_000 + frame_id * 7_000_000,
                    capture_wait_ms=6.5,
                    image=None,
                )
            )
            self._condition.notify_all()

    def read(self):
        with self._condition:
            if not self._frames:
                self._condition.wait(0.01)
            if not self._frames:
                return None
            return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True
        with self._condition:
            self._condition.notify_all()


def _service(
    cfg: RuntimeConfig | None = None,
    *,
    capability_runner=lambda device: CAPS_TEXT,
    source_factory=lambda profile: FakeSource(),
    empty_read_sleep_s: float | None = None,
) -> CaptureService:
    config = cfg or RuntimeConfig()
    kwargs = {
        "config": config.capture,
        "capability_runner": capability_runner,
        "source_factory": source_factory,
    }
    if empty_read_sleep_s is not None:
        kwargs["empty_read_sleep_s"] = empty_read_sleep_s
    service = CaptureService(**kwargs)
    _SERVICES.append(service)
    return service


@pytest.fixture(autouse=True)
def _stop_services_after_test():
    yield
    while _SERVICES:
        service = _SERVICES.pop()
        service.stop("test cleanup")


def _wait_until(predicate, timeout_s: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_configure_selects_profile_from_caps() -> None:
    service = _service()

    state = service.configure("/dev/video0")

    assert state.available is True
    assert state.profile.pixel_format == "MJPG"
    assert state.profile.fps == 144
    assert state.backend == "gst:test"


def test_configure_starts_capture_session_and_publishes_frames() -> None:
    source = FakeSource()
    service = _service(source_factory=lambda profile: source)

    state = service.configure("/dev/video0")
    frame = service.wait_preview_frame(after_frame_id=0, timeout_s=0.2)
    service.stop("test complete")

    assert state.available is True
    assert frame is not None
    assert source.count >= 1
    assert source.closed is True


def test_preview_fps_uses_sliding_window(monkeypatch) -> None:
    service = _service()
    frame = object()
    ticks = iter(
        [
            1_000_000_000,
            1_200_000_000,
            1_500_000_000,
            1_900_000_000,
            2_000_000_000,
        ]
    )
    monkeypatch.setattr("novasight.capture.service.time.monotonic_ns", lambda: next(ticks))

    for _ in range(5):
        service.record_preview_output(frame, target_fps=30)

    assert service.state.preview_fps == pytest.approx(4.0)


def test_service_exposes_session_running_state_after_configure() -> None:
    service = _service()

    state = service.configure("/dev/video0")

    assert state.available is True
    assert service.source is not None
    assert service.session.running is True

    service.stop("test complete")


def test_default_source_factory_prefers_native_appsink(monkeypatch) -> None:
    from novasight.capture import service as service_module

    attempts: list[str] = []

    class FakeAppSinkSource:
        backend_label = "gst-appsink:test"

        def __init__(self, profile, candidate) -> None:
            attempts.append(candidate.label)
            self.backend_label = candidate.label

        def opened_and_readable(self) -> bool:
            return self.backend_label == "gst-appsink:nvmm-mjpg-ioauto"

        def read(self):
            return None

        def close(self) -> None:
            pass

    monkeypatch.setattr(service_module, "GstAppSinkFrameSource", FakeAppSinkSource)

    source = service_module._open_default_source(service_module.select_capture_profile(
        "/dev/video0",
        service_module.query_capabilities("/dev/video0", runner=lambda device: CAPS_TEXT).capabilities,
    ))

    assert source.backend_label == "gst-appsink:nvmm-mjpg-ioauto"
    assert attempts[:3] == [
        "gst-appsink:nvmm-mjpg-iomode2",
        "gst-appsink:nvmm-mjpg-iomode4",
        "gst-appsink:nvmm-mjpg-ioauto",
    ]


def test_default_source_factory_passes_roi_size_to_appsink_candidates(monkeypatch) -> None:
    from novasight.capture import service as service_module

    pipelines: list[str] = []

    class FakeAppSinkSource:
        def __init__(self, profile, candidate) -> None:
            del profile
            pipelines.append(candidate.pipeline)
            self.backend_label = candidate.label

        def opened_and_readable(self) -> bool:
            return True

        def read(self):
            return None

        def close(self) -> None:
            pass

    monkeypatch.setattr(service_module, "GstAppSinkFrameSource", FakeAppSinkSource)

    service_module._open_default_source(
        service_module.select_capture_profile(
            "/dev/video0",
            service_module.query_capabilities("/dev/video0", runner=lambda device: CAPS_TEXT).capabilities,
        ),
        roi_size=320,
    )

    assert pipelines
    assert "left=800 right=1120 top=380 bottom=700" in pipelines[0]
    assert "src-crop" not in pipelines[0]


def test_default_source_factory_keeps_roi_on_python_appsink_path(monkeypatch) -> None:
    from novasight.capture import service as service_module

    attempts: list[str] = []

    class FakeAppSinkSource:
        def __init__(self, profile, candidate) -> None:
            del profile
            attempts.append(candidate.pipeline)
            self.backend_label = candidate.label

        def opened_and_readable(self) -> bool:
            return False

        def read(self):
            return None

        def close(self) -> None:
            pass

    monkeypatch.setattr(service_module, "GstAppSinkFrameSource", FakeAppSinkSource)

    try:
        service_module._open_default_source(
            service_module.select_capture_profile(
                "/dev/video0",
                service_module.query_capabilities("/dev/video0", runner=lambda device: CAPS_TEXT).capabilities,
            ),
            roi_size=320,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected all fake GPU candidates to fail")

    assert attempts
    assert all("left=800 right=1120 top=380 bottom=700" in attempt for attempt in attempts)
    assert all("src-crop" not in attempt for attempt in attempts)
    assert all("video/x-raw,format=BGRx,width=320,height=320 ! appsink" in attempt for attempt in attempts)
    assert all("width=1920,height=1080 ! appsink" not in attempt for attempt in attempts)


def test_default_source_factory_opens_selected_appsink_only_once(monkeypatch) -> None:
    from novasight.capture import service as service_module

    opened: list[str] = []

    class FakeAppSinkSource:
        def __init__(self, profile, candidate) -> None:
            opened.append(candidate.label)
            self.backend_label = candidate.label
            self.closed = False

        @classmethod
        def probe(cls, profile, candidate):
            raise AssertionError("default source factory should not probe by double-opening")

        def opened_and_readable(self) -> bool:
            return True

        def read(self):
            return None

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(service_module, "GstAppSinkFrameSource", FakeAppSinkSource)

    source = service_module._open_default_source(service_module.select_capture_profile(
        "/dev/video0",
        service_module.query_capabilities("/dev/video0", runner=lambda device: CAPS_TEXT).capabilities,
    ))

    assert source.backend_label == "gst-appsink:nvmm-mjpg-iomode2"
    assert opened == ["gst-appsink:nvmm-mjpg-iomode2"]


def test_default_source_factory_reports_appsink_failures_without_opencv(monkeypatch) -> None:
    from novasight.capture import service as service_module

    attempts: list[str] = []

    class FailingAppSinkSource:
        def __init__(self, profile, candidate) -> None:
            del profile
            attempts.append(candidate.label)
            self.backend_label = candidate.label

        def opened_and_readable(self) -> bool:
            return False

        def close(self) -> None:
            pass

    monkeypatch.setattr(service_module, "GstAppSinkFrameSource", FailingAppSinkSource)

    profile = service_module.select_capture_profile(
        "/dev/video0",
        service_module.query_capabilities("/dev/video0", runner=lambda device: CAPS_TEXT).capabilities,
    )

    with pytest.raises(RuntimeError) as exc:
        service_module._open_default_source(profile)

    message = str(exc.value)
    assert "no capture backend opened for /dev/video0" in message
    assert "gst-appsink:nvmm-mjpg-iomode2" in message
    assert "opencv" not in message.lower()
    assert all(label.startswith("gst-appsink:") for label in attempts)


def test_wait_preview_frame_updates_stream_diagnostics() -> None:
    service = _service()
    service.configure("/dev/video0")

    first = service.wait_preview_frame(after_frame_id=0, timeout_s=0.2)
    assert first is not None
    second = service.wait_preview_frame(after_frame_id=first.frame_id, timeout_s=0.2)

    assert second is not None
    assert service.state.available is True
    assert service.state.capture_wait_ms == 6.5
    assert service.state.frame_period_ms > 0
    assert service.state.fps_capture > 0
    assert service.state.last_error is None


def test_latest_preview_frame_proxies_session_latest_frame() -> None:
    service = _service()
    service.configure("/dev/video0")

    frame = service.wait_preview_frame(after_frame_id=0, timeout_s=0.2)
    preview = service.get_latest_preview_frame()

    assert preview is frame
    assert service.source.count >= 1


def test_wait_preview_frame_returns_only_newer_frame() -> None:
    source = ControlledSource()
    service = _service(source_factory=lambda profile: source)
    service.configure("/dev/video0")
    source.emit(1)
    first = service.wait_preview_frame(after_frame_id=0, timeout_s=0.2)
    assert first is not None

    assert service.wait_preview_frame(after_frame_id=0, timeout_s=0) is first
    assert service.wait_preview_frame(after_frame_id=first.frame_id, timeout_s=0) is None


def test_state_reflects_async_session_failure_without_explicit_sync() -> None:
    service = _service(source_factory=lambda profile: ExplodingSource())
    service.configure("/dev/video0")

    assert _wait_until(lambda: service.state.last_error is not None)
    assert service.state.available is False
    assert service.source is None
    assert "camera disconnected" in str(service.state.last_error)


def test_reconfigure_closes_previous_source() -> None:
    sources: list[FakeSource] = []

    def factory(profile):
        source = FakeSource()
        sources.append(source)
        return source

    service = _service(source_factory=factory)
    service.configure("/dev/video0")
    service.configure("/dev/video1")

    assert len(sources) == 2
    assert sources[0].closed is True
    assert sources[1].closed is False
    assert service.source is sources[1]


def test_reconfigure_same_device_closes_previous_source_before_opening_next() -> None:
    sources: list[FakeSource] = []
    second_open_saw_first_closed: list[bool] = []

    def factory(profile):
        if sources:
            second_open_saw_first_closed.append(sources[0].closed)
        source = FakeSource()
        sources.append(source)
        return source

    service = _service(
        capability_runner=lambda device: MULTI_CAPS_TEXT,
        source_factory=factory,
    )
    service.configure(
        "/dev/video0",
        preference="manual",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=60,
    )

    service.configure(
        "/dev/video0",
        preference="manual",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=120,
    )

    assert second_open_saw_first_closed == [True]
    assert sources[0].closed is True
    assert sources[1].closed is False


def test_configure_restarts_session_when_profile_is_unchanged() -> None:
    sources: list[FakeSource] = []

    def factory(profile):
        source = FakeSource()
        sources.append(source)
        return source

    service = _service(
        capability_runner=lambda device: MULTI_CAPS_TEXT,
        source_factory=factory,
    )
    first = service.configure(
        "/dev/video0",
        preference="manual",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=120,
    )
    second = service.configure(
        "/dev/video0",
        preference="manual",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=120,
    )

    assert len(sources) == 2
    assert sources[0].closed is True
    assert sources[1].closed is False
    assert second is not first


def test_failed_reconfigure_stops_previous_source_and_reports_unavailable() -> None:
    service = _service(
        capability_runner=lambda device: CAPS_TEXT if device == "/dev/video0" else None
    )
    service.configure("/dev/video0")

    state = service.configure("/dev/missing")

    assert service.source is None
    assert state is service.state
    assert state.available is False
    assert state.device == "/dev/missing"
    assert service.state.last_error is not None


def test_blank_device_does_not_fallback_to_default() -> None:
    cfg = RuntimeConfig()
    cfg.capture.device = "/dev/video0"
    queried: list[str] = []

    def runner(device: str) -> str | None:
        queried.append(device)
        return None

    service = _service(cfg=cfg, capability_runner=runner)

    state = service.configure("")

    assert queried == []
    assert state.available is False
    assert state.device == ""
    assert state.last_error == "capture device is required"
    assert service.source is None


def test_blank_device_after_success_stops_existing_source() -> None:
    sources: list[FakeSource] = []

    def factory(profile):
        source = FakeSource()
        sources.append(source)
        return source

    service = _service(source_factory=factory)
    service.configure("/dev/video0")

    state = service.configure("")

    assert sources[0].closed is True
    assert service.source is None
    assert state is service.state
    assert state.available is False
    assert service.last_config_error is not None
    assert service.last_config_error.available is False
    assert "required" in str(service.last_config_error.last_error)


def test_source_factory_error_stops_previous_source() -> None:
    cfg = RuntimeConfig()
    old_source = FakeSource()

    def factory(profile):
        if profile.device == "/dev/video0":
            return old_source
        raise RuntimeError("backend failed to open")

    service = _service(cfg=cfg, source_factory=factory)
    service.configure("/dev/video0")

    state = service.configure("/dev/video1")

    assert old_source.closed is True
    assert service.source is None
    assert state is service.state
    assert state.available is False
    assert state.device == "/dev/video1"
    assert state.last_error == "backend failed to open"


def test_same_device_reconfigure_failure_does_not_keep_closed_source() -> None:
    sources: list[FakeSource] = []

    def factory(profile):
        if not sources:
            source = FakeSource()
            sources.append(source)
            return source
        raise RuntimeError("backend failed to open")

    service = _service(
        capability_runner=lambda device: MULTI_CAPS_TEXT,
        source_factory=factory,
    )
    service.configure(
        "/dev/video0",
        preference="manual",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=60,
    )

    state = service.configure(
        "/dev/video0",
        preference="manual",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=120,
    )

    assert sources[0].closed is True
    assert service.source is None
    assert state.available is False
    assert "backend failed to open" in str(state.last_error)


def test_incomplete_doctor_command_returns_nonzero_without_starting_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from novasight import main as novasight_main

    def fail_run(*args, **kwargs):
        raise AssertionError("uvicorn.run should not be called")

    monkeypatch.setattr(novasight_main.uvicorn, "run", fail_run)

    result = novasight_main.main(["doctor"])

    assert result != 0
