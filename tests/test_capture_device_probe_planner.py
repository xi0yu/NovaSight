from __future__ import annotations

from fractions import Fraction

import pytest

from novasight.capture.device_probe import DeviceCapability, DeviceProbe
from novasight.capture.pipeline_planner import (
    InfeasibleConfiguration,
    PipelinePlanner,
    RoiConfig,
)


V4L2_ALL_OUTPUT = """
Driver Info:
        Driver name      : uvcvideo
        Card type        : GC553G2
        Bus info         : usb-0000:01:00.0-2
"""


V4L2_FORMATS_OUTPUT = """
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture

        [0]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 1920x1080
                        Interval: Discrete 0.008s (120.000 fps)
                        Interval: Discrete 0.017s (60.000 fps)
                Size: Discrete 2560x1440
                        Interval: Discrete 0.017s (60.000 fps)
        [1]: 'NV12' (Y/CbCr 4:2:0)
                Size: Discrete 1920x1080
                        Interval: Discrete 0.017s (60.000 fps)
        [2]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 1920x1080
                        Interval: Discrete 0.033s (30.000 fps)
"""


def test_device_probe_parses_v4l2_device_metadata_and_formats(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def run(command: list[str]) -> str:
        calls.append(tuple(command))
        if command[-1] == "--all":
            return V4L2_ALL_OUTPUT
        if command[-1] == "--list-formats-ext":
            return V4L2_FORMATS_OUTPUT
        raise AssertionError(command)

    monkeypatch.setattr(DeviceProbe, "_run_command", staticmethod(run))

    capability = DeviceProbe.probe_device("/dev/video0")

    assert capability.device_path == "/dev/video0"
    assert capability.driver == "uvcvideo"
    assert capability.card_name == "GC553G2"
    assert capability.bus_info == "usb-0000:01:00.0-2"
    assert capability.pixel_formats[0].format == "MJPG"
    assert capability.pixel_formats[0].description == "Motion-JPEG, compressed"
    assert capability.pixel_formats[0].resolutions[0].fps_list == [
        Fraction(120, 1),
        Fraction(60, 1),
    ]
    assert calls == [
        ("v4l2-ctl", "-d", "/dev/video0", "--all"),
        ("v4l2-ctl", "-d", "/dev/video0", "--list-formats-ext"),
    ]


def test_device_probe_all_uses_list_devices_and_skips_failed_devices(monkeypatch) -> None:
    def run(command: list[str]) -> str:
        if command == ["v4l2-ctl", "--list-devices"]:
            return "GC553G2:\n\t/dev/video0\n\t/dev/video1\n"
        if command[2] == "/dev/video0" and command[-1] == "--all":
            return V4L2_ALL_OUTPUT
        if command[2] == "/dev/video0" and command[-1] == "--list-formats-ext":
            return V4L2_FORMATS_OUTPUT
        raise RuntimeError("device busy")

    monkeypatch.setattr(DeviceProbe, "_run_command", staticmethod(run))

    capabilities = DeviceProbe.probe_all()

    assert [cap.device_path for cap in capabilities] == ["/dev/video0"]


def test_pipeline_planner_prefers_raw_nv12_when_available() -> None:
    plan = PipelinePlanner.plan(
        capabilities=[_capability()],
        device="/dev/video0",
        target_res=(1920, 1080),
        target_fps=Fraction(60, 1),
        model_input_size=(640, 640),
        model_hash="abc123",
        roi_config=RoiConfig(size=640, offset_x=0, offset_y=0),
        validator=lambda _plan: True,
        cache_dir=None,
    )

    assert plan.capture_format == "NV12"
    assert plan.decode_backend == "none"
    assert plan.memory_domain == "NVMM"
    assert plan.crop_rect is not None
    assert plan.crop_rect.width == 640
    assert "video/x-raw,format=NV12,width=1920,height=1080,framerate=60/1" in (
        plan.generate_gst_launch_string()
    )


def test_pipeline_planner_falls_back_when_candidate_validation_fails() -> None:
    attempted: list[str] = []

    def validator(plan) -> bool:
        attempted.append(plan.capture_format)
        return plan.capture_format != "NV12"

    plan = PipelinePlanner.plan(
        capabilities=[_capability()],
        device="/dev/video0",
        target_res=(1920, 1080),
        target_fps=Fraction(60, 1),
        model_input_size=(640, 640),
        model_hash="abc123",
        roi_config=RoiConfig(size=640, offset_x=0, offset_y=0),
        validator=validator,
        cache_dir=None,
    )

    assert attempted[:2] == ["NV12", "MJPG"]
    assert plan.capture_format == "MJPG"
    assert plan.decode_backend == "nvv4l2decoder"


def test_pipeline_planner_reuses_cached_plan_for_identical_inputs(tmp_path) -> None:
    first = PipelinePlanner.plan(
        capabilities=[_capability()],
        device="/dev/video0",
        target_res=(1920, 1080),
        target_fps=Fraction(60, 1),
        model_input_size=(640, 640),
        model_hash="abc123",
        roi_config=RoiConfig(size=640, offset_x=0, offset_y=0),
        cache_dir=tmp_path,
    )

    second = PipelinePlanner.plan(
        capabilities=[],
        device="/dev/video0",
        target_res=(1920, 1080),
        target_fps=Fraction(60, 1),
        model_input_size=(640, 640),
        model_hash="abc123",
        roi_config=RoiConfig(size=640, offset_x=0, offset_y=0),
        cache_dir=tmp_path,
    )

    assert second == first
    assert list(tmp_path.glob("*.json"))


def test_pipeline_planner_raises_for_unavailable_target() -> None:
    with pytest.raises(InfeasibleConfiguration):
        PipelinePlanner.plan(
            capabilities=[_capability()],
            device="/dev/video0",
            target_res=(3840, 2160),
            target_fps=Fraction(60, 1),
            model_input_size=(640, 640),
            model_hash="abc123",
            roi_config=RoiConfig(size=640, offset_x=0, offset_y=0),
            cache_dir=None,
        )


def _capability() -> DeviceCapability:
    return DeviceProbe.parse_device_outputs(
        "/dev/video0",
        all_output=V4L2_ALL_OUTPUT,
        formats_output=V4L2_FORMATS_OUTPUT,
    )
