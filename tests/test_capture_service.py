from novasight.capture.service import CaptureService
from novasight.config import RuntimeConfig


class FakeSource:
    backend_label = "gst:test"

    def __init__(self) -> None:
        self.count = 0

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
        pass


def test_capture_service_selects_profile_from_config() -> None:
    cfg = RuntimeConfig()
    service = CaptureService(
        config=cfg.capture,
        capability_runner=lambda device: "[0]: 'MJPG' (Motion-JPEG)\n    Size: Discrete 1920x1080\n        Interval: Discrete 0.007s (144.000 fps)\n",
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
