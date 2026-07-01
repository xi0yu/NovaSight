from novasight.capture import CaptureProfile
from novasight.capture.pipeline import build_pipeline_candidates, select_open_source


def _profile(fmt: str = "MJPG") -> CaptureProfile:
    return CaptureProfile(
        device="/dev/video0",
        pixel_format=fmt,
        width=1920,
        height=1080,
        fps=144 if fmt == "MJPG" else 60,
        preference="auto_high_fps",
        selection_reason="test",
    )


def test_mjpg_candidates_start_with_nvmm_decoder() -> None:
    candidates = build_pipeline_candidates(_profile("MJPG"))

    assert candidates[0].label == "gst:nvmm-mjpg-iomode2"
    assert "v4l2src device=/dev/video0" in candidates[0].pipeline
    assert "image/jpeg,width=1920,height=1080,framerate=144/1" in candidates[0].pipeline
    assert "nvv4l2decoder mjpeg=1" in candidates[0].pipeline
    assert candidates[-1].label == "opencv:v4l2"


def test_nv12_candidates_start_with_nvmm_nv12() -> None:
    candidates = build_pipeline_candidates(_profile("NV12"))

    assert candidates[0].label == "gst:nvmm-nv12-iomode2"
    assert (
        "video/x-raw,format=NV12,width=1920,height=1080,framerate=60/1"
        in candidates[0].pipeline
    )


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
