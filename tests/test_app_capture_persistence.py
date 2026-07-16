from __future__ import annotations

from types import SimpleNamespace

from novasight.api.app import _auto_restore_capture, _load_active_model
from novasight.config import RuntimeConfig


def test_auto_restore_skips_when_source_default_is_null() -> None:
    cfg = RuntimeConfig()
    cfg.source.default = "null"
    cfg.capture.device = "/dev/video0"

    called = []

    class FakeCapture:
        def configure_profile_only(self, *args, **kwargs):
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
        def configure_profile_only(self, *args, **kwargs):
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
        def configure_profile_only(self, device, **kwargs):
            captured_kwargs.append({"device": device, **kwargs})
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


def test_active_deepstream_engine_does_not_require_legacy_model_profile(tmp_path) -> None:
    engine_path = tmp_path / "active.engine"
    engine_path.write_bytes(b"engine")
    config = RuntimeConfig()
    config.inference.backend = "deepstream_nvinfer"
    events: list[tuple[str, str]] = []
    deployment = SimpleNamespace(artifact_id=7)
    artifact = SimpleNamespace(id=7, kind="engine", version_id=11)
    version = SimpleNamespace(id=11, project_id=13)
    project = SimpleNamespace(id=13)
    models = SimpleNamespace(
        get_active_deployment=lambda: deployment,
        get_artifact=lambda artifact_id: artifact if artifact_id == 7 else None,
        get_version=lambda version_id: version if version_id == 11 else None,
        get_project=lambda project_id: project if project_id == 13 else None,
        resolve_artifact_path=lambda _artifact: engine_path,
    )
    inference = SimpleNamespace(
        unload=lambda reason: events.append(("unload", reason)),
        disable=lambda reason: events.append(("disable", reason)),
    )

    _load_active_model(models, inference, config)

    assert events == [
        ("unload", "TensorRT engine ownership delegated to DeepStream nvinfer")
    ]
