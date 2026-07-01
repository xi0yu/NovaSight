from __future__ import annotations

import time
from collections.abc import Callable

from novasight.config.runtime import CaptureConfig

from .caps import query_capabilities, run_v4l2_ctl
from .pipeline import CaptureCandidate, select_open_source
from .profile import select_capture_profile
from .source import FrameSource, OpenCvFrameSource
from .state import CaptureCapabilities, CaptureProfile, CaptureRuntimeState


def _open_default_source(profile: CaptureProfile) -> FrameSource:
    selected = select_open_source(
        profile,
        lambda candidate: OpenCvFrameSource.probe(profile, candidate),
    )
    selected_candidate = CaptureCandidate(
        label=selected.label,
        pipeline=selected.pipeline,
    )
    return OpenCvFrameSource(profile, selected_candidate)


class CaptureService:
    def __init__(
        self,
        config: CaptureConfig,
        *,
        capability_runner: Callable[[str], str | None] | None = None,
        source_factory: Callable[[CaptureProfile], FrameSource] | None = None,
        empty_read_sleep_s: float = 0.001,
    ) -> None:
        self.config = config
        self.capability_runner = capability_runner
        self.source_factory = source_factory or _open_default_source
        self.empty_read_sleep_s = empty_read_sleep_s
        self.state = CaptureRuntimeState(device=config.device)
        self.source: FrameSource | None = None

    def capabilities(self, device: str = "/dev/video0") -> CaptureCapabilities:
        return query_capabilities(
            device,
            runner=self.capability_runner or run_v4l2_ctl,
        )

    def configure(
        self,
        device: str | None = None,
        *,
        preference: str | None = None,
        pixel_format: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> CaptureRuntimeState:
        selected_device = self.config.device if device is None else device
        if not selected_device.strip():
            self.state = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error="capture device is required",
            )
            return self.state
        selected_preference = preference if preference is not None else self.config.preference
        selected_pixel_format = (
            pixel_format if pixel_format is not None else self.config.pixel_format
        )
        selected_width = width if width is not None else self.config.width
        selected_height = height if height is not None else self.config.height
        selected_fps = fps if fps is not None else self.config.fps
        if self.source is not None:
            self.source.close()
            self.source = None
        caps = self.capabilities(selected_device)
        if not caps.available:
            self.state = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error=caps.reason,
            )
            return self.state
        try:
            profile = select_capture_profile(
                selected_device,
                caps.capabilities,
                selected_preference,  # type: ignore[arg-type]
                pixel_format=selected_pixel_format or None,
                width=selected_width or None,
                height=selected_height or None,
                fps=selected_fps or None,
            )
            source = self.source_factory(profile)
        except Exception as exc:
            self.source = None
            self.state = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error=str(exc),
            )
            return self.state
        self.source = source
        self.config.preference = selected_preference
        self.config.pixel_format = selected_pixel_format
        self.config.width = selected_width
        self.config.height = selected_height
        self.config.fps = selected_fps
        self.state = CaptureRuntimeState(
            available=True,
            device=selected_device,
            profile=profile,
            backend=source.backend_label,
            last_error=None,
        )
        return self.state

    def capture_frames(
        self,
        *,
        seconds: float | None = None,
        max_frames: int | None = None,
        max_empty_reads: int | None = None,
    ) -> CaptureRuntimeState:
        if self.source is None:
            raise RuntimeError("capture source is not configured")
        start = time.monotonic()
        previous_ts: int | None = None
        count = 0
        empty_reads = 0
        empty_read_limit = max_empty_reads
        if empty_read_limit is None and seconds is None and max_frames is not None:
            empty_read_limit = 100
        while True:
            if max_frames is not None and count >= max_frames:
                break
            if seconds is not None and time.monotonic() - start >= seconds:
                break
            frame = self.source.read()
            if frame is None:
                self.state.frames_dropped += 1
                empty_reads += 1
                if empty_read_limit is not None and empty_reads >= empty_read_limit:
                    break
                if self.empty_read_sleep_s > 0:
                    time.sleep(self.empty_read_sleep_s)
                continue
            empty_reads = 0
            count += 1
            self.state.capture_wait_ms = frame.capture_wait_ms
            if previous_ts is not None:
                self.state.frame_period_ms = (frame.ts_ns - previous_ts) / 1e6
            previous_ts = frame.ts_ns
        elapsed = max(time.monotonic() - start, 0.000001)
        self.state.fps_capture = count / elapsed
        return self.state
