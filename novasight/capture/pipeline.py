from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .state import CaptureProfile


@dataclass(frozen=True)
class CaptureCandidate:
    label: str
    pipeline: str


@dataclass(frozen=True)
class SelectedCaptureBackend:
    label: str
    pipeline: str
    failures: list[str] = field(default_factory=list)


def build_pipeline_candidates(profile: CaptureProfile) -> list[CaptureCandidate]:
    device = profile.device
    width = profile.width
    height = profile.height
    fps = profile.fps
    fmt = profile.pixel_format.upper()
    sink = "appsink drop=true max-buffers=1 sync=false"

    mjpg_caps = f"image/jpeg,width={width},height={height},framerate={fps}/1"
    nv12_caps = f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1"
    yuyv_caps = f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1"

    def gst(label: str, body: str) -> CaptureCandidate:
        return CaptureCandidate(label=label, pipeline=body)

    bgrx_to_sink = (
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        f"{sink}"
    )
    mjpg = [
        gst(
            "gst:nvmm-mjpg-iomode2",
            f"v4l2src device={device} io-mode=2 ! {mjpg_caps} ! "
            f"jpegparse ! nvv4l2decoder mjpeg=1 ! {bgrx_to_sink}",
        ),
        gst(
            "gst:nvmm-mjpg-iomode4",
            f"v4l2src device={device} io-mode=4 ! {mjpg_caps} ! "
            f"jpegparse ! nvv4l2decoder mjpeg=1 ! {bgrx_to_sink}",
        ),
        gst(
            "gst:nvmm-mjpg-ioauto",
            f"v4l2src device={device} ! {mjpg_caps} ! "
            f"jpegparse ! nvv4l2decoder mjpeg=1 ! {bgrx_to_sink}",
        ),
        gst(
            "gst:cpu-jpegdec-mjpg",
            f"v4l2src device={device} ! {mjpg_caps} ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! {sink}",
        ),
    ]
    nv12 = [
        gst(
            "gst:nvmm-nv12-iomode2",
            f"v4l2src device={device} io-mode=2 ! {nv12_caps} ! {bgrx_to_sink}",
        ),
        gst(
            "gst:nvmm-nv12-iomode4",
            f"v4l2src device={device} io-mode=4 ! {nv12_caps} ! {bgrx_to_sink}",
        ),
        gst(
            "gst:nvmm-nv12-ioauto",
            f"v4l2src device={device} ! {nv12_caps} ! {bgrx_to_sink}",
        ),
        gst(
            "gst:cpu-nv12-videoconvert",
            f"v4l2src device={device} ! {nv12_caps} ! "
            f"videoconvert ! video/x-raw,format=BGR ! {sink}",
        ),
    ]
    yuyv = [
        gst(
            "gst:nvmm-yuyv-iomode2",
            f"v4l2src device={device} io-mode=2 ! {yuyv_caps} ! {bgrx_to_sink}",
        ),
        gst(
            "gst:cpu-yuyv-videoconvert",
            f"v4l2src device={device} ! {yuyv_caps} ! "
            f"videoconvert ! video/x-raw,format=BGR ! {sink}",
        ),
    ]

    ordered = mjpg + nv12 + yuyv
    if fmt == "NV12":
        ordered = nv12 + yuyv + mjpg
    elif fmt == "YUYV":
        ordered = yuyv + nv12 + mjpg
    return ordered + [CaptureCandidate(label="opencv:v4l2", pipeline="")]


def select_open_source(
    profile: CaptureProfile,
    opener: Callable[[CaptureCandidate], bool],
) -> SelectedCaptureBackend:
    failures: list[str] = []
    for candidate in build_pipeline_candidates(profile):
        try:
            opened = opener(candidate)
        except Exception as exc:
            failures.append(f"{candidate.label}: {exc}")
            continue
        if opened:
            return SelectedCaptureBackend(
                label=candidate.label,
                pipeline=candidate.pipeline,
                failures=failures,
            )
        failures.append(candidate.label)
    raise RuntimeError(
        f"no capture backend opened for {profile.device}: {', '.join(failures)}"
    )
