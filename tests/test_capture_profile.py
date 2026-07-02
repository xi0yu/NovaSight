import pytest

from novasight.capture import CaptureCapability
from novasight.capture.profile import select_capture_profile


CAPS = [
    CaptureCapability("MJPG", 1920, 1080, [240, 120, 60]),
    CaptureCapability("MJPG", 1920, 1080, [144, 120, 60]),
    CaptureCapability("NV12", 1920, 1080, [60]),
    CaptureCapability("YUYV", 1280, 720, [60]),
    CaptureCapability("MJPG", 3840, 2160, [30]),
]


def test_auto_high_fps_prefers_1080p_120_mjpg_for_jetson_nvmm_route() -> None:
    profile = select_capture_profile("/dev/video0", CAPS, "auto_high_fps")

    assert profile.pixel_format == "MJPG"
    assert profile.width == 1920
    assert profile.height == 1080
    assert profile.fps == 120
    assert "jetson nvmm" in profile.selection_reason.lower()


def test_auto_low_latency_prefers_nv12_when_fps_is_usable() -> None:
    profile = select_capture_profile("/dev/video0", CAPS, "auto_low_latency")

    assert profile.pixel_format == "NV12"
    assert profile.width == 1920
    assert profile.height == 1080
    assert profile.fps == 60


def test_auto_balanced_prefers_1080p_before_4k30() -> None:
    profile = select_capture_profile("/dev/video0", CAPS, "auto_balanced")

    assert profile.width == 1920
    assert profile.height == 1080
    assert profile.fps >= 60


def test_manual_requires_exact_supported_profile() -> None:
    profile = select_capture_profile(
        "/dev/video1",
        CAPS,
        "manual",
        pixel_format="NV12",
        width=1920,
        height=1080,
        fps=60,
    )

    assert profile.device == "/dev/video1"
    assert profile.pixel_format == "NV12"


def test_manual_rejects_unsupported_profile() -> None:
    with pytest.raises(ValueError, match="unsupported capture profile"):
        select_capture_profile(
            "/dev/video0",
            CAPS,
            "manual",
            pixel_format="NV12",
            width=1920,
            height=1080,
            fps=144,
        )


def test_unknown_preference_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown capture preference"):
        select_capture_profile("/dev/video0", CAPS, "auto_fast")
