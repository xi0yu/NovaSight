"""Tests for the capture service state machine.

Focuses on the recovery contract: bounded retries on empty reads and read
exceptions, close-error handling, and the no-mutation guarantee on
unsuccessful reconfigures. Mechanical details like every single empty-read
backoff step are folded into one test.
"""
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
    return CaptureService(**kwargs)


def test_configure_selects_profile_from_caps() -> None:
    service = _service()

    state = service.configure("/dev/video0")

    assert state.available is True
    assert state.profile.pixel_format == "MJPG"
    assert state.profile.fps == 144
    assert state.backend == "gst:test"


def test_default_source_factory_prefers_native_appsink(monkeypatch) -> None:
    from novasight.capture import service as service_module

    attempts: list[str] = []

    class FakeAppSinkSource:
        backend_label = "gst-appsink:test"

        def __init__(self, profile, candidate) -> None:
            attempts.append(candidate.label)
            self.backend_label = candidate.label

        def opened_and_readable(self) -> bool:
            return self.backend_label == "gst-appsink:nvmm-mjpg-iomode2"

        def read(self):
            return None

        def close(self) -> None:
            pass

    class FailingOpenCvSource:
        @classmethod
        def probe(cls, profile, candidate):
            raise AssertionError("opencv fallback should not run when appsink opens")

    monkeypatch.setattr(service_module, "GstAppSinkFrameSource", FakeAppSinkSource)
    monkeypatch.setattr(service_module, "OpenCvFrameSource", FailingOpenCvSource)

    source = service_module._open_default_source(service_module.select_capture_profile(
        "/dev/video0",
        service_module.query_capabilities("/dev/video0", runner=lambda device: CAPS_TEXT).capabilities,
    ))

    assert source.backend_label == "gst-appsink:nvmm-mjpg-iomode2"
    assert attempts[0] == "gst-appsink:nvmm-mjpg-iomode2"


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

    class FailingOpenCvSource:
        def __init__(self, profile, candidate) -> None:
            raise AssertionError("opencv fallback should not run when appsink opens")

    monkeypatch.setattr(service_module, "GstAppSinkFrameSource", FakeAppSinkSource)
    monkeypatch.setattr(service_module, "OpenCvFrameSource", FailingOpenCvSource)

    source = service_module._open_default_source(service_module.select_capture_profile(
        "/dev/video0",
        service_module.query_capabilities("/dev/video0", runner=lambda device: CAPS_TEXT).capabilities,
    ))

    assert source.backend_label == "gst-appsink:nvmm-mjpg-iomode2"
    assert opened == ["gst-appsink:nvmm-mjpg-iomode2"]


def test_read_frame_updates_stream_diagnostics() -> None:
    service = _service()
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


def test_configure_reuses_existing_source_when_profile_is_unchanged() -> None:
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

    assert len(sources) == 1
    assert sources[0].closed is False
    assert second is first


def test_failed_reconfigure_keeps_previous_source() -> None:
    service = _service(
        capability_runner=lambda device: CAPS_TEXT if device == "/dev/video0" else None
    )
    service.configure("/dev/video0")

    state = service.configure("/dev/missing")

    assert service.source is not None
    assert state is service.state
    assert state.available is True
    assert state.device == "/dev/video0"
    assert service.state.last_error is None


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


def test_blank_device_after_success_keeps_existing_source() -> None:
    sources: list[FakeSource] = []

    def factory(profile):
        source = FakeSource()
        sources.append(source)
        return source

    service = _service(source_factory=factory)
    service.configure("/dev/video0")

    state = service.configure("")

    assert sources[0].closed is False
    assert service.source is sources[0]
    assert state is service.state
    assert state.available is True
    assert service.last_config_error is not None
    assert service.last_config_error.available is False
    assert "required" in str(service.last_config_error.last_error)


def test_source_factory_error_keeps_previous_source() -> None:
    cfg = RuntimeConfig()
    old_source = FakeSource()

    def factory(profile):
        if profile.device == "/dev/video0":
            return old_source
        raise RuntimeError("backend failed to open")

    service = _service(cfg=cfg, source_factory=factory)
    service.configure("/dev/video0")

    state = service.configure("/dev/video1")

    assert old_source.closed is False
    assert service.source is old_source
    assert state is service.state
    assert state.available is True
    assert state.device == "/dev/video0"
    assert state.last_error is None


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


def test_empty_reads_eventually_mark_source_unavailable() -> None:
    sources: list[EmptySource] = []

    def factory(profile):
        source = EmptySource()
        sources.append(source)
        return source

    service = _service(
        source_factory=factory,
        empty_read_sleep_s=0,
    )
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_empty_reads=3, max_recoveries=1)

    assert len(sources) == 2
    assert all(source.closed for source in sources)
    assert service.source is None
    assert state.available is False
    assert state.last_error == "capture produced 3 empty reads"
    assert state.recoveries == 1


@pytest.mark.parametrize(
    "scenario",
    [
        "close_failed_then_succeed",
        "read_then_close_failed",
    ],
)
def test_close_errors_during_recovery_appear_in_last_error(scenario: str) -> None:
    sources: list = []

    def factory(profile):
        if not sources:
            source = (
                CloseExplodingSource()
                if scenario == "close_failed_then_succeed"
                else ReadAndCloseExplodingSource()
            )
        else:
            source = EmptySource()
        sources.append(source)
        return source

    service = _service(source_factory=factory, empty_read_sleep_s=0)
    service.configure("/dev/video0")

    if scenario == "close_failed_then_succeed":
        state = service.capture_frames(max_frames=1, max_empty_reads=1, max_recoveries=1)
        assert "close failed" in str(state.last_error)
    else:
        state = service.capture_frames(max_frames=1, max_recoveries=1)
        assert "read failed" in str(state.last_error)
        assert "close failed" in str(state.last_error)

    assert len(sources) == 1
    assert all(source.closed for source in sources)
    assert service.source is None
    assert state.available is False


def test_read_exception_attempts_bounded_recovery() -> None:
    sources: list[ExplodingSource] = []

    def factory(profile):
        source = ExplodingSource()
        sources.append(source)
        return source

    service = _service(source_factory=factory, empty_read_sleep_s=0)
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_recoveries=1)

    assert len(sources) == 2
    assert all(source.closed for source in sources)
    assert service.source is None
    assert state.available is False
    assert state.frames_dropped == 2
    assert state.recoveries == 1
    assert "camera disconnected" in str(state.last_error)


def test_mark_unavailable_handles_close_errors() -> None:
    source = CloseExplodingSource()
    service = _service(source_factory=lambda profile: source, empty_read_sleep_s=0)
    service.configure("/dev/video0")

    state = service.capture_frames(max_frames=1, max_empty_reads=1, max_recoveries=0)

    assert source.closed is True
    assert service.source is None
    assert state.available is False
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
