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


def build_appsink_candidates(profile: CaptureProfile) -> list[CaptureCandidate]:
    device = profile.device
    width = profile.width
    height = profile.height
    fps = profile.fps
    fmt = profile.pixel_format.upper()
    output_width = width
    output_height = height
    sink = "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"

    mjpg_caps = f"image/jpeg,width={width},height={height},framerate={fps}/1"
    nv12_caps = f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1"
    yuyv_caps = f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1"

    def gst(label: str, body: str) -> CaptureCandidate:
        return CaptureCandidate(label=f"gst-appsink:{label}", pipeline=body)

    nvmm_caps = (
        "video/x-raw(memory:NVMM),format=NV12,"
        f"width={output_width},height={output_height}"
    )
    cpu_bgr_caps = f"video/x-raw,format=BGRx,width={output_width},height={output_height}"
    mjpg_nvmm_tail = f"jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! {nvmm_caps} ! {sink}"
    nv12_nvmm_tail = f"nvvidconv ! {nvmm_caps} ! {sink}"
    yuyv_nvmm_tail = f"nvvidconv ! {nvmm_caps} ! {sink}"
    mjpg_cpu_tail = f"jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! {cpu_bgr_caps} ! {sink}"
    nv12_cpu_tail = f"nvvidconv ! {cpu_bgr_caps} ! {sink}"
    yuyv_cpu_tail = f"nvvidconv ! {cpu_bgr_caps} ! {sink}"

    mjpg = [
        gst(
            "nvmm-mjpg-iomode2",
            f"v4l2src device={device} io-mode=2 ! {mjpg_caps} ! {mjpg_nvmm_tail}",
        ),
        gst(
            "nvmm-mjpg-iomode4",
            f"v4l2src device={device} io-mode=4 ! {mjpg_caps} ! {mjpg_nvmm_tail}",
        ),
        gst(
            "nvmm-mjpg-ioauto",
            f"v4l2src device={device} ! {mjpg_caps} ! {mjpg_nvmm_tail}",
        ),
        gst(
            "cpu-bgr-mjpg-iomode2",
            f"v4l2src device={device} io-mode=2 ! {mjpg_caps} ! {mjpg_cpu_tail}",
        ),
        gst(
            "cpu-bgr-mjpg-ioauto",
            f"v4l2src device={device} ! {mjpg_caps} ! {mjpg_cpu_tail}",
        ),
    ]
    nv12 = [
        gst(
            "nvmm-nv12-iomode2",
            f"v4l2src device={device} io-mode=2 ! {nv12_caps} ! {nv12_nvmm_tail}",
        ),
        gst(
            "nvmm-nv12-iomode4",
            f"v4l2src device={device} io-mode=4 ! {nv12_caps} ! {nv12_nvmm_tail}",
        ),
        gst(
            "nvmm-nv12-ioauto",
            f"v4l2src device={device} ! {nv12_caps} ! {nv12_nvmm_tail}",
        ),
        gst(
            "cpu-bgr-nv12-iomode2",
            f"v4l2src device={device} io-mode=2 ! {nv12_caps} ! {nv12_cpu_tail}",
        ),
        gst(
            "cpu-bgr-nv12-ioauto",
            f"v4l2src device={device} ! {nv12_caps} ! {nv12_cpu_tail}",
        ),
    ]
    yuyv = [
        gst(
            "nvmm-yuyv-iomode2",
            f"v4l2src device={device} io-mode=2 ! {yuyv_caps} ! {yuyv_nvmm_tail}",
        ),
        gst(
            "nvmm-yuyv-iomode4",
            f"v4l2src device={device} io-mode=4 ! {yuyv_caps} ! {yuyv_nvmm_tail}",
        ),
        gst(
            "nvmm-yuyv-ioauto",
            f"v4l2src device={device} ! {yuyv_caps} ! {yuyv_nvmm_tail}",
        ),
        gst(
            "cpu-bgr-yuyv-iomode2",
            f"v4l2src device={device} io-mode=2 ! {yuyv_caps} ! {yuyv_cpu_tail}",
        ),
        gst(
            "cpu-bgr-yuyv-ioauto",
            f"v4l2src device={device} ! {yuyv_caps} ! {yuyv_cpu_tail}",
        ),
    ]

    ordered = mjpg + nv12 + yuyv
    if fmt == "NV12":
        ordered = nv12 + yuyv + mjpg
    elif fmt == "YUYV":
        ordered = yuyv + nv12 + mjpg
    return ordered


def select_open_source(
    profile: CaptureProfile,
    opener: Callable[[CaptureCandidate], bool],
    candidates: list[CaptureCandidate] | None = None,
) -> SelectedCaptureBackend:
    failures: list[str] = []
    for candidate in candidates or build_pipeline_candidates(profile):
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
