from novasight.capture import CaptureCapability, query_capabilities
from novasight.capture.caps import parse_v4l2_formats


V4L2_TEXT = """
[0]: 'MJPG' (Motion-JPEG, compressed)
    Size: Discrete 1920x1080
        Interval: Discrete 0.007s (144.000 fps)
        Interval: Discrete 0.008s (120.000 fps)
        Interval: Discrete 0.017s (60.000 fps)
    Size: Discrete 1280x720
        Interval: Discrete 0.007s (144.000 fps)
[1]: 'NV12' (Y/CbCr 4:2:0)
    Size: Discrete 1920x1080
        Interval: Discrete 0.017s (60.000 fps)
[2]: 'YUYV' (YUYV 4:2:2)
    Size: Discrete 1280x720
        Interval: Discrete 0.033s (30.000 fps)
"""


def test_parse_v4l2_formats_groups_format_size_and_fps() -> None:
    caps = parse_v4l2_formats(V4L2_TEXT)

    assert caps == [
        CaptureCapability("MJPG", 1920, 1080, [144, 120, 60]),
        CaptureCapability("MJPG", 1280, 720, [144]),
        CaptureCapability("NV12", 1920, 1080, [60]),
        CaptureCapability("YUYV", 1280, 720, [30]),
    ]


def test_query_capabilities_reports_unavailable_when_runner_fails() -> None:
    result = query_capabilities("/dev/video0", runner=lambda device: None)

    assert result.available is False
    assert result.device == "/dev/video0"
    assert result.capabilities == []
    assert "v4l2-ctl" in result.reason


def test_query_capabilities_uses_injected_runner() -> None:
    seen: list[str] = []

    def runner(device: str) -> str:
        seen.append(device)
        return V4L2_TEXT

    result = query_capabilities("/dev/video2", runner=runner)

    assert seen == ["/dev/video2"]
    assert result.available is True
    assert result.capabilities[0].fps_list == [144, 120, 60]
