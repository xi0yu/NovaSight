import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np

from novasight.capture import CaptureProfile
from novasight.capture.pipeline import (
    CaptureCandidate,
    build_appsink_candidates,
    build_pipeline_candidates,
    select_open_source,
)
from novasight.capture.source import GstAppSinkFrameSource, OpenCvFrameSource
from novasight.capture.source import _sample_to_bgr


def _profile(fmt: str = "MJPG") -> CaptureProfile:
    return CaptureProfile(
        device="/dev/video0",
        pixel_format=fmt,
        width=1920,
        height=1080,
        fps=120 if fmt == "MJPG" else 60,
        preference="auto_high_fps",
        selection_reason="test",
    )


def test_mjpg_candidates_start_with_nvmm_decoder() -> None:
    candidates = build_pipeline_candidates(_profile("MJPG"))

    assert candidates[0].label == "gst:nvmm-mjpg-iomode2"
    assert "v4l2src device=/dev/video0" in candidates[0].pipeline
    assert "image/jpeg,width=1920,height=1080,framerate=120/1" in candidates[0].pipeline
    assert "nvv4l2decoder mjpeg=1" in candidates[0].pipeline
    assert candidates[-1].label == "opencv:v4l2"


def test_nv12_candidates_start_with_nvmm_nv12() -> None:
    candidates = build_pipeline_candidates(_profile("NV12"))

    assert candidates[0].label == "gst:nvmm-nv12-iomode2"
    assert (
        "video/x-raw,format=NV12,width=1920,height=1080,framerate=60/1"
        in candidates[0].pipeline
    )


def test_appsink_candidates_do_not_use_opencv_labels() -> None:
    candidates = build_appsink_candidates(_profile("MJPG"))

    assert candidates[0].label == "gst-appsink:cpu-bgr-mjpg-iomode2"
    assert "appsink name=sink" in candidates[0].pipeline
    assert "video/x-raw,format=BGRx" in candidates[0].pipeline
    assert "nvv4l2decoder mjpeg=1" in candidates[0].pipeline
    assert any(
        "video/x-raw(memory:NVMM),format=NV12,width=320,height=320"
        in candidate.pipeline
        for candidate in candidates
    )
    assert all(not candidate.label.startswith("opencv:") for candidate in candidates)


def test_appsink_candidates_map_yuyv_to_gstreamer_yuy2_then_nv12() -> None:
    candidates = build_appsink_candidates(_profile("YUYV"))

    assert candidates[0].label == "gst-appsink:cpu-bgr-yuyv-iomode2"
    assert "video/x-raw,format=YUY2,width=1920,height=1080" in candidates[0].pipeline
    assert "video/x-raw,format=BGRx" in candidates[0].pipeline


def test_select_open_source_returns_first_candidate_that_reads() -> None:
    attempts: list[str] = []

    def opener(candidate):
        attempts.append(candidate.label)
        return candidate.label == "gst:cpu-jpegdec-mjpg"

    selected = select_open_source(_profile("MJPG"), opener=opener)

    assert selected.label == "gst:cpu-jpegdec-mjpg"
    assert attempts[:4] == [
        "gst:nvmm-mjpg-iomode2",
        "gst:nvmm-mjpg-iomode4",
        "gst:nvmm-mjpg-ioauto",
        "gst:cpu-jpegdec-mjpg",
    ]
    assert selected.failures == attempts[:3]


def test_select_open_source_raises_when_all_candidates_fail() -> None:
    try:
        select_open_source(_profile("MJPG"), opener=lambda candidate: False)
    except RuntimeError as exc:
        assert "no capture backend opened" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_select_open_source_records_opener_exception_and_continues() -> None:
    attempts: list[str] = []

    def opener(candidate):
        attempts.append(candidate.label)
        if candidate.label == "gst:nvmm-mjpg-iomode2":
            raise RuntimeError("decoder unavailable")
        return candidate.label == "gst:cpu-jpegdec-mjpg"

    selected = select_open_source(_profile("MJPG"), opener=opener)

    assert selected.label == "gst:cpu-jpegdec-mjpg"
    assert attempts[:4] == [
        "gst:nvmm-mjpg-iomode2",
        "gst:nvmm-mjpg-iomode4",
        "gst:nvmm-mjpg-ioauto",
        "gst:cpu-jpegdec-mjpg",
    ]
    assert selected.failures[0] == "gst:nvmm-mjpg-iomode2: decoder unavailable"
    assert selected.failures[1:3] == [
        "gst:nvmm-mjpg-iomode4",
        "gst:nvmm-mjpg-ioauto",
    ]


def test_opencv_probe_closes_source_when_first_read_succeeds(monkeypatch) -> None:
    released: list[bool] = []

    class FakeVideoCapture:
        def __init__(self, *_args) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def read(self):
            return True, object()

        def release(self) -> None:
            self.released = True
            released.append(True)

    fake_cv2 = SimpleNamespace(
        CAP_GSTREAMER=1800,
        VideoCapture=FakeVideoCapture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    opened = OpenCvFrameSource.probe(
        _profile(),
        CaptureCandidate(label="gst:test", pipeline="pipeline"),
    )

    assert opened is True
    assert released == [True]


def test_opencv_probe_closes_source_when_first_read_fails(monkeypatch) -> None:
    released: list[bool] = []

    class FakeVideoCapture:
        def __init__(self, *_args) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def read(self):
            return False, None

        def release(self) -> None:
            self.released = True
            released.append(True)

    fake_cv2 = SimpleNamespace(
        CAP_GSTREAMER=1800,
        VideoCapture=FakeVideoCapture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    opened = OpenCvFrameSource.probe(
        _profile(),
        CaptureCandidate(label="gst:test", pipeline="pipeline"),
    )

    assert opened is False
    assert released == [True]


def test_gstreamer_sample_to_bgr_converts_nv12_buffer() -> None:
    class FakeStructure:
        def get_int(self, name: str):
            return True, 2

        def get_string(self, name: str):
            return "NV12"

    class FakeCaps:
        def get_size(self) -> int:
            return 1

        def get_structure(self, index: int):
            return FakeStructure()

    class FakeBuffer:
        def __init__(self) -> None:
            self.unmapped = False

        def map(self, flags):
            return True, SimpleNamespace(data=bytes([16, 16, 16, 16, 128, 128]))

        def unmap(self, info) -> None:
            self.unmapped = True

    class FakeSample:
        def __init__(self) -> None:
            self.buffer = FakeBuffer()

        def get_caps(self):
            return FakeCaps()

        def get_buffer(self):
            return self.buffer

    sample = FakeSample()
    image, width, height = _sample_to_bgr(
        sample, SimpleNamespace(MapFlags=SimpleNamespace(READ=1))
    )

    assert (width, height) == (2, 2)
    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert int(image.max()) == 0
    assert sample.buffer.unmapped is True


def test_gstreamer_appsink_close_waits_for_null_state() -> None:
    calls: list[tuple[str, object | None]] = []

    class FakePipeline:
        def set_state(self, state) -> None:
            calls.append(("set_state", state))

        def get_state(self, timeout) -> None:
            calls.append(("get_state", timeout))

    fake_gst = SimpleNamespace(State=SimpleNamespace(NULL="NULL"), SECOND=1_000_000_000)
    source = object.__new__(GstAppSinkFrameSource)
    source._closed = False
    source._pipeline = FakePipeline()
    source._Gst = fake_gst

    source.close()

    assert calls == [
        ("set_state", "NULL"),
        ("get_state", 2_000_000_000),
    ]


def test_gstreamer_appsink_constructor_closes_pipeline_when_first_frame_fails(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object | None]] = []

    class FakeStructure:
        def get_int(self, name: str):
            return True, 2

        def get_string(self, name: str):
            return "NVMM"

    class FakeCaps:
        def get_size(self) -> int:
            return 1

        def get_structure(self, index: int):
            return FakeStructure()

    class FakeBuffer:
        def map(self, flags):
            return True, SimpleNamespace(data=bytes([0, 0, 0, 0]))

        def unmap(self, info) -> None:
            pass

    class FakeSample:
        def get_caps(self):
            return FakeCaps()

        def get_buffer(self):
            return FakeBuffer()

    class FakeAppSink:
        def try_pull_sample(self, timeout):
            return FakeSample()

    class FakePipeline:
        def get_by_name(self, name: str):
            return FakeAppSink()

        def set_state(self, state) -> str:
            calls.append(("set_state", state))
            return "SUCCESS"

        def get_state(self, timeout) -> None:
            calls.append(("get_state", timeout))

    fake_gst = SimpleNamespace(
        SECOND=1_000_000_000,
        MapFlags=SimpleNamespace(READ=1),
        State=SimpleNamespace(PLAYING="PLAYING", NULL="NULL"),
        StateChangeReturn=SimpleNamespace(FAILURE="FAILURE"),
        is_initialized=lambda: True,
        init=lambda args: None,
        parse_launch=lambda pipeline: FakePipeline(),
    )
    gi = ModuleType("gi")
    gi.require_version = lambda *args: None
    repository = ModuleType("gi.repository")
    repository.Gst = fake_gst
    repository.GstApp = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)

    try:
        GstAppSinkFrameSource(
            _profile(),
            CaptureCandidate(label="gst-appsink:test", pipeline="pipeline"),
        )
    except RuntimeError as exc:
        assert "unsupported GStreamer sample format" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert calls == [
        ("set_state", "PLAYING"),
        ("set_state", "NULL"),
        ("get_state", 2_000_000_000),
    ]
