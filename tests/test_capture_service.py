import pytest

from novasight.capture.service import CaptureService
from novasight.config import RuntimeConfig


CAPS_TEXT = "[0]: 'MJPG' (Motion-JPEG)\n    Size: Discrete 1920x1080\n        Interval: Discrete 0.007s (144.000 fps)\n"


class FakeSource:
    backend_label = "gst:test"

    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def read(self):
        from novasight.capture.source import CapturedFrame

        self.count += 1
        return CapturedFrame(
            frame_id=self.count,
            width=1920,
            height=1080,
            pixel_format="BGR",
            ts_ns=1_000_000_000 + self.count * 7_000_000,
            capture_wait_ms=6.5,
            image=None,
        )

    def close(self) -> None:
        self.closed = True


class EmptySource:
    backend_label = "gst:empty"

    def __init__(self) -> None:
        self.closed = False

    def read(self):
        return None

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


class CloseExplodingSource(EmptySource):
    backend_label = "gst:close-broken"

    def close(self) -> None:
        self.closed = True
        raise RuntimeError("close failed")


class ReadAndCloseExplodingSource(CloseExplodingSource):
    def read(self):
        raise RuntimeError("read failed")


def test_capture_service_selects_profile_from_config() -> None:
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: FakeSource(),
    )

    state = service.configure("/dev/video0")

    assert state.available is True
    assert state.profile is not None
    assert state.profile.pixel_format == "MJPG"
    assert state.profile.fps == 144
    assert state.backend == "gst:test"


def test_capture_service_smoke_updates_timing() -> None:
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: "[0]: 'NV12' (Y/CbCr)\n    Size: Discrete 1920x1080\n        Interval: Discrete 0.017s (60.000 fps)\n",
        source_factory=lambda profile: FakeSource(),
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=3)

    assert state.fps_capture > 0
    assert state.capture_wait_ms == 6.5
    assert state.frame_period_ms > 0
    assert state.last_error is None


def test_capture_service_read_frame_updates_stream_diagnostics() -> None:
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: FakeSource(),
    )
    service.configure("/dev/video0")

    first = service.read_frame()
    second = service.read_frame()

    assert first is not None
    assert second is not None
    assert service.state.available is True
    assert service.state.capture_wait_ms == 6.5
    assert service.state.frame_period_ms > 0
    assert service.state.fps_capture > 0
    assert service.state.last_error is None


def test_capture_service_reconfigure_closes_previous_source() -> None:
    cfg = RuntimeConfig()
    sources: list[FakeSource] = []

    def source_factory(profile):
        source = FakeSource()
        sources.append(source)
        return source

    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=source_factory,
    )

    first_state = service.configure("/dev/video0")
    second_state = service.configure("/dev/video1")

    assert first_state.available is True
    assert second_state.available is True
    assert len(sources) == 2
    assert sources[0].closed is True
    assert sources[1].closed is False
    assert service.source is sources[1]


def test_capture_service_failed_reconfigure_keeps_previous_source() -> None:
    cfg = RuntimeConfig()
    source = FakeSource()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT if device == "/dev/video0" else None,
        source_factory=lambda profile: source,
    )
    service.configure("/dev/video0")

    state = service.configure("/dev/missing")

    assert source.closed is False
    assert service.source is source
    assert state is service.state
    assert state.available is True
    assert state.device == "/dev/video0"
    assert service.state.last_error is None


def test_capture_service_blank_device_does_not_fallback_to_default() -> None:
    cfg = RuntimeConfig()
    cfg.capture.device = "/dev/video0"
    queried_devices: list[str] = []

    def capability_runner(device: str) -> str | None:
        queried_devices.append(device)
        return None

    service = CaptureService(
        config=cfg.capture,
        capability_runner=capability_runner,
        source_factory=lambda profile: FakeSource(),
    )

    state = service.configure("")

    assert queried_devices == []
    assert state.available is False
    assert state.device == ""
    assert state.last_error == "capture device is required"
    assert service.source is None


def test_capture_service_blank_device_keeps_existing_source() -> None:
    cfg = RuntimeConfig()
    sources: list[FakeSource] = []

    def source_factory(profile):
        source = FakeSource()
        sources.append(source)
        return source

    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=source_factory,
    )
    first_state = service.configure("/dev/video0")

    state = service.configure("")

    assert first_state.available is True
    assert sources[0].closed is False
    assert service.source is sources[0]
    assert state is service.state
    assert state.available is True
    assert service.last_config_error is not None
    assert service.last_config_error.available is False
    assert "required" in str(service.last_config_error.last_error)


def test_capture_service_source_factory_error_clears_state() -> None:
    cfg = RuntimeConfig()
    old_source = FakeSource()
    source_error = "backend failed to open"

    def source_factory(profile):
        if profile.device == "/dev/video0":
            return old_source
        raise RuntimeError(source_error)

    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=source_factory,
    )
    first_state = service.configure("/dev/video0")

    state = service.configure("/dev/video1")

    assert first_state.available is True
    assert old_source.closed is False
    assert service.source is old_source
    assert state is service.state
    assert state.available is True
    assert state.device == "/dev/video0"
    assert state.last_error is None


def test_capture_service_empty_reads_back_off_and_terminate() -> None:
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: EmptySource(),
        empty_read_sleep_s=0.001,
    )
    service.configure("/dev/video0")

    state = service.capture_frames(seconds=0.01)

    assert state.frames_dropped > 0
    assert state.frames_dropped < 100
    assert state.fps_capture == 0


def test_capture_service_empty_read_limit_attempts_bounded_recovery() -> None:
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: EmptySource(),
        empty_read_sleep_s=0,
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_empty_reads=3)

    assert state.frames_dropped == 6
    assert state.recoveries == 1
    assert state.available is False
    assert service.source is None


def test_capture_service_empty_read_recovery_failure_marks_unavailable() -> None:
    cfg = RuntimeConfig()
    sources: list[EmptySource] = []

    def source_factory(profile):
        source = EmptySource()
        sources.append(source)
        return source

    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=source_factory,
        empty_read_sleep_s=0,
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_empty_reads=3, max_recoveries=1)

    assert len(sources) == 2
    assert all(source.closed for source in sources)
    assert service.source is None
    assert state.available is False
    assert state.frames_dropped == 6
    assert state.recoveries == 1
    assert state.last_error == "capture produced 3 empty reads"


def test_capture_service_recovery_handles_close_errors() -> None:
    cfg = RuntimeConfig()
    sources: list[EmptySource] = []

    def source_factory(profile):
        if not sources:
            source = CloseExplodingSource()
        else:
            source = EmptySource()
        sources.append(source)
        return source

    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=source_factory,
        empty_read_sleep_s=0,
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_empty_reads=1, max_recoveries=1)

    assert len(sources) == 2
    assert all(source.closed for source in sources)
    assert service.source is None
    assert state.available is False
    assert "close failed" in str(state.last_error)


def test_capture_service_read_exception_attempts_bounded_recovery() -> None:
    cfg = RuntimeConfig()
    sources: list[ExplodingSource] = []

    def source_factory(profile):
        source = ExplodingSource()
        sources.append(source)
        return source

    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=source_factory,
        empty_read_sleep_s=0,
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_recoveries=1)

    assert len(sources) == 2
    assert all(source.closed for source in sources)
    assert service.source is None
    assert state.available is False
    assert state.frames_dropped == 2
    assert state.recoveries == 1
    assert "camera disconnected" in str(state.last_error)


def test_capture_service_mark_unavailable_handles_close_errors() -> None:
    cfg = RuntimeConfig()
    source = CloseExplodingSource()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=lambda profile: source,
        empty_read_sleep_s=0,
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_empty_reads=1, max_recoveries=0)

    assert source.closed is True
    assert service.source is None
    assert state.available is False
    assert "close failed" in str(state.last_error)


def test_capture_service_read_recovery_preserves_close_error() -> None:
    cfg = RuntimeConfig()
    sources: list[EmptySource] = []

    def source_factory(profile):
        if not sources:
            source = ReadAndCloseExplodingSource()
        else:
            source = EmptySource()
        sources.append(source)
        return source

    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: CAPS_TEXT,
        source_factory=source_factory,
        empty_read_sleep_s=0,
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_recoveries=1)

    assert len(sources) == 2
    assert all(source.closed for source in sources)
    assert service.source is None
    assert state.available is False
    assert "read failed" in str(state.last_error)
    assert "close failed" in str(state.last_error)


def test_incomplete_doctor_command_returns_nonzero_without_starting_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from novasight import main as novasight_main

    def fail_run(*args, **kwargs):
        raise AssertionError("uvicorn.run should not be called")

    monkeypatch.setattr(novasight_main.uvicorn, "run", fail_run)

    result = novasight_main.main(["doctor"])

    assert result != 0
