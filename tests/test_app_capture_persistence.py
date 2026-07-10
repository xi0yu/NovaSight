from __future__ import annotations

from types import SimpleNamespace

from novasight.api.app import _auto_restore_capture
from novasight.config import RuntimeConfig


def test_auto_restore_skips_when_source_default_is_null() -> None:
    cfg = RuntimeConfig()
    cfg.source.default = "null"
    cfg.capture.device = "/dev/video0"

    called = []

    class FakeCapture:
        def configure(self, *args, **kwargs):
            called.append((args, kwargs))
            return SimpleNamespace(available=False, last_error="not invoked", profile=None)

    _auto_restore_capture(FakeCapture(), cfg)
    assert called == []


def test_auto_restore_skips_when_no_device() -> None:
    cfg = RuntimeConfig()
    cfg.source.default = "capture"
    cfg.capture.device = ""

    called = []

    class FakeCapture:
        def configure(self, *args, **kwargs):
            called.append((args, kwargs))
            return SimpleNamespace(available=False, last_error="", profile=None)

    _auto_restore_capture(FakeCapture(), cfg)
    assert called == []


def test_auto_restore_opens_capture_when_source_is_capture() -> None:
    cfg = RuntimeConfig()
    cfg.source.default = "capture"
    cfg.capture.device = "/dev/video0"
    cfg.capture.preference = "manual"
    cfg.capture.pixel_format = "MJPG"
    cfg.capture.width = 2560
    cfg.capture.height = 1440
    cfg.capture.fps = 120

    captured_kwargs = []

    class FakeCapture:
        def configure(self, **kwargs):
            captured_kwargs.append(kwargs)
            profile = SimpleNamespace(
                pixel_format="MJPG", width=2560, height=1440, fps=120
            )
            return SimpleNamespace(available=True, profile=profile, last_error="")

    _auto_restore_capture(FakeCapture(), cfg)

    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    assert kwargs["device"] == "/dev/video0"
    assert kwargs["preference"] == "manual"
    assert kwargs["pixel_format"] == "MJPG"
    assert kwargs["width"] == 2560
    assert kwargs["height"] == 1440
    assert kwargs["fps"] == 120
