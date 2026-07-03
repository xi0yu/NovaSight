from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from novasight.roi import center_roi_region

from .state import CaptureProfile

@dataclass(frozen=True)
class CaptureCandidate:
    label: str
    pipeline: str
    source_width: int | None = None
    source_height: int | None = None
    roi_size: int | None = None
    roi_offset_x: int = 0
    roi_offset_y: int = 0


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

    nvmm_to_sink = f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! {sink}"
    mjpg = [
        gst(
            "gst:nvmm-mjpg-iomode2",
            f"v4l2src device={device} io-mode=2 ! {mjpg_caps} ! "
            f"jpegparse ! nvv4l2decoder mjpeg=1 ! {nvmm_to_sink}",
        ),
        gst(
            "gst:nvmm-mjpg-iomode4",
            f"v4l2src device={device} io-mode=4 ! {mjpg_caps} ! "
            f"jpegparse ! nvv4l2decoder mjpeg=1 ! {nvmm_to_sink}",
        ),
        gst(
            "gst:nvmm-mjpg-ioauto",
            f"v4l2src device={device} ! {mjpg_caps} ! "
            f"jpegparse ! nvv4l2decoder mjpeg=1 ! {nvmm_to_sink}",
        ),
    ]
    nv12 = [
        gst(
            "gst:nvmm-nv12-iomode2",
            f"v4l2src device={device} io-mode=2 ! {nv12_caps} ! {nvmm_to_sink}",
        ),
        gst(
            "gst:nvmm-nv12-iomode4",
            f"v4l2src device={device} io-mode=4 ! {nv12_caps} ! {nvmm_to_sink}",
        ),
        gst(
            "gst:nvmm-nv12-ioauto",
            f"v4l2src device={device} ! {nv12_caps} ! {nvmm_to_sink}",
        ),
    ]
    yuyv = [
        gst(
            "gst:nvmm-yuyv-iomode2",
            f"v4l2src device={device} io-mode=2 ! {yuyv_caps} ! {nvmm_to_sink}",
        ),
    ]

    ordered = mjpg + nv12 + yuyv
    if fmt == "NV12":
        ordered = nv12 + yuyv + mjpg
    elif fmt == "YUYV":
        ordered = yuyv + nv12 + mjpg
    return ordered


def build_appsink_candidates(
    profile: CaptureProfile,
    *,
    roi_size: int | None = None,
) -> list[CaptureCandidate]:
    device = profile.device
    width = profile.width
    height = profile.height
    fps = profile.fps
    fmt = profile.pixel_format.upper()
    crop_properties = ""
    candidate_source_width: int | None = None
    candidate_source_height: int | None = None
    candidate_roi_size: int | None = None
    candidate_roi_offset_x = 0
    candidate_roi_offset_y = 0
    if roi_size is not None:
        crop_x, crop_y, crop_size = center_roi_region(
            source_width=width,
            source_height=height,
            requested_size=roi_size,
        )
        output_width = crop_size
        output_height = crop_size
        crop_right = crop_x + crop_size
        crop_bottom = crop_y + crop_size
        crop_properties = (
            f" left={crop_x} right={crop_right} top={crop_y} bottom={crop_bottom}"
        )
        candidate_source_width = width
        candidate_source_height = height
        candidate_roi_size = crop_size
        candidate_roi_offset_x = crop_x
        candidate_roi_offset_y = crop_y
    else:
        output_width = width
        output_height = height
    sink = "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"

    mjpg_caps = f"image/jpeg,width={width},height={height},framerate={fps}/1"
    nv12_caps = f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1"
    yuyv_caps = f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1"

    def gst(label: str, body: str) -> CaptureCandidate:
        return CaptureCandidate(
            label=f"gst-appsink:{label}",
            pipeline=body,
            source_width=candidate_source_width,
            source_height=candidate_source_height,
            roi_size=candidate_roi_size,
            roi_offset_x=candidate_roi_offset_x,
            roi_offset_y=candidate_roi_offset_y,
        )

    appsink_caps = f"video/x-raw,format=BGRx,width={output_width},height={output_height}"
    nvvidconv = f"nvvidconv{crop_properties}"
    mjpg_appsink_tail = f"jpegparse ! nvv4l2decoder mjpeg=1 ! {nvvidconv} ! {appsink_caps} ! {sink}"
    nv12_appsink_tail = f"{nvvidconv} ! {appsink_caps} ! {sink}"
    yuyv_appsink_tail = f"{nvvidconv} ! {appsink_caps} ! {sink}"

    mjpg = [
        gst(
            "nvmm-mjpg-iomode2",
            f"v4l2src device={device} io-mode=2 ! {mjpg_caps} ! {mjpg_appsink_tail}",
        ),
        gst(
            "nvmm-mjpg-iomode4",
            f"v4l2src device={device} io-mode=4 ! {mjpg_caps} ! {mjpg_appsink_tail}",
        ),
        gst(
            "nvmm-mjpg-ioauto",
            f"v4l2src device={device} ! {mjpg_caps} ! {mjpg_appsink_tail}",
        ),
    ]
    nv12 = [
        gst(
            "nvmm-nv12-iomode2",
            f"v4l2src device={device} io-mode=2 ! {nv12_caps} ! {nv12_appsink_tail}",
        ),
        gst(
            "nvmm-nv12-iomode4",
            f"v4l2src device={device} io-mode=4 ! {nv12_caps} ! {nv12_appsink_tail}",
        ),
        gst(
            "nvmm-nv12-ioauto",
            f"v4l2src device={device} ! {nv12_caps} ! {nv12_appsink_tail}",
        ),
    ]
    yuyv = [
        gst(
            "nvmm-yuyv-iomode2",
            f"v4l2src device={device} io-mode=2 ! {yuyv_caps} ! {yuyv_appsink_tail}",
        ),
        gst(
            "nvmm-yuyv-iomode4",
            f"v4l2src device={device} io-mode=4 ! {yuyv_caps} ! {yuyv_appsink_tail}",
        ),
        gst(
            "nvmm-yuyv-ioauto",
            f"v4l2src device={device} ! {yuyv_caps} ! {yuyv_appsink_tail}",
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
