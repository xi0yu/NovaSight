import numpy as np

from novasight.api.routes_capture import _deepstream_mjpeg_frames, _mjpeg_frames
from novasight.capture.preview import render_preview_frame
from novasight.capture.source import CapturedFrame, FrameResource, GstResourceFrameSource
from novasight.capture.state import CaptureRuntimeState


class _PreviewSink:
    def __init__(self, sample):
        self.sample = sample

    def try_pull_sample(self, _timeout_ns: int):
        sample = self.sample
        self.sample = None
        return sample


class _PreviewCapture:
    def __init__(self, frame: CapturedFrame):
        self.frame = frame
        self.state = CaptureRuntimeState(available=True)

    def wait_preview_frame(self, *, after_frame_id=None, timeout_s=0.0):
        del after_frame_id, timeout_s
        return self.frame

    def record_preview_drop(self, *, target_fps: int) -> None:
        del target_fps

    def record_preview_output(self, frame: CapturedFrame, *, target_fps: int) -> None:
        del frame, target_fps


class _DeepStreamPreviewBackend:
    running = True

    def __init__(self):
        self.calls = 0

    def wait_preview_jpeg(self, *, after_sequence=None, timeout_s=0.0):
        del after_sequence, timeout_s
        self.calls += 1
        return (7, b"\xff\xd8preview\xff\xd9")


def test_preview_renders_cpu_snapshot_attached_to_nvmm_frame() -> None:
    preview_image = np.zeros((320, 320, 3), dtype=np.uint8)
    preview_image[:, :, 1] = 180
    resource = FrameResource(
        kind="gstreamer_sample",
        handle=object(),
        memory="nvmm",
        width=320,
        height=320,
        pixel_format="NV12",
        source="appsink",
    )
    frame = CapturedFrame(
        frame_id=9,
        width=320,
        height=320,
        pixel_format="NV12",
        ts_ns=456,
        capture_wait_ms=0.8,
        image=None,
        frame_resource=resource,
        preview_image=preview_image,
        source_width=1920,
        source_height=1080,
        roi_size=320,
        roi_offset_x=800,
        roi_offset_y=380,
    )

    rendered = render_preview_frame(frame, roi_size=320)

    assert rendered.size == (320, 320)
    assert rendered.getpixel((10, 10)) == (0, 180, 0)


def test_nvmm_source_keeps_preview_sample_lazy() -> None:
    sample = object()
    source = object.__new__(GstResourceFrameSource)
    source._preview_appsink = _PreviewSink(sample)
    source._latest_preview_resource = None

    resource = source._pull_latest_preview_resource(width=320, height=320)

    assert resource is not None
    assert resource.kind == "gstreamer_preview_sample"
    assert resource.handle is sample
    assert resource.memory == "cpu"
    assert source._pull_latest_preview_resource(width=320, height=320) is resource


def test_nvmm_preview_snapshot_reaches_mjpeg_stream() -> None:
    preview_image = np.zeros((32, 32, 3), dtype=np.uint8)
    preview_image[:, :, 2] = 220
    frame = CapturedFrame(
        frame_id=4,
        width=32,
        height=32,
        pixel_format="NV12",
        ts_ns=456,
        capture_wait_ms=0.4,
        image=None,
        preview_image=preview_image,
        roi_size=32,
    )
    capture = _PreviewCapture(frame)

    chunks = list(
        _mjpeg_frames(
            capture,
            preview_fps=60,
            roi_size=32,
            max_frames=1,
            max_attempts=1,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert capture.state.preview_available is True
    assert capture.state.preview_reason == ""


def test_deepstream_hardware_jpeg_reaches_mjpeg_stream_without_reencoding() -> None:
    backend = _DeepStreamPreviewBackend()

    chunks = list(
        _deepstream_mjpeg_frames(
            backend,
            preview_fps=30,
            max_frames=1,
            max_attempts=1,
        )
    )

    assert len(chunks) == 1
    assert b"Content-Length: 11" in chunks[0]
    assert b"\xff\xd8preview\xff\xd9" in chunks[0]


def test_paused_deepstream_preview_stream_stops_without_pulling_jpeg() -> None:
    backend = _DeepStreamPreviewBackend()
    backend.preview_active = False

    chunks = list(
        _deepstream_mjpeg_frames(
            backend,
            preview_fps=30,
            max_frames=1,
            max_attempts=1,
        )
    )

    assert chunks == []
    assert backend.calls == 0
