from __future__ import annotations

import time
from collections.abc import Callable

from novasight.config.runtime import CaptureConfig

from .caps import query_capabilities, run_v4l2_ctl
from .pipeline import CaptureCandidate, select_open_source
from .profile import select_capture_profile
from .source import CapturedFrame, FrameSource, OpenCvFrameSource
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
        self.last_config_error: CaptureRuntimeState | None = None
        self._last_frame_ts_ns: int | None = None

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
            failure = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error="capture device is required",
            )
            self.last_config_error = failure
            if self.source is None:
                self.state = failure
                return self.state
            return self.state
        selected_preference = preference if preference is not None else self.config.preference
        selected_pixel_format = (
            pixel_format if pixel_format is not None else self.config.pixel_format
        )
        selected_width = width if width is not None else self.config.width
        selected_height = height if height is not None else self.config.height
        selected_fps = fps if fps is not None else self.config.fps
        caps = self.capabilities(selected_device)
        if not caps.available:
            failure = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error=caps.reason,
            )
            self.last_config_error = failure
            if self.source is None:
                self.state = failure
                return self.state
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
            failure = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error=str(exc),
            )
            self.last_config_error = failure
            if self.source is None:
                self.state = failure
                return self.state
            return self.state
        old_source = self.source
        self.source = source
        if old_source is not None:
            old_source.close()
        self.config.device = selected_device
        self.config.preference = selected_preference
        self.config.pixel_format = selected_pixel_format
        self.config.width = selected_width
        self.config.height = selected_height
        self.config.fps = selected_fps
        self.last_config_error = None
        self.state = CaptureRuntimeState(
            available=True,
            device=selected_device,
            profile=profile,
            backend=source.backend_label,
            last_error=None,
        )
        self._last_frame_ts_ns = None
        return self.state

    def read_frame(self) -> CapturedFrame | None:
        if self.source is None:
            self.state.last_error = "capture source is not configured"
            raise RuntimeError(self.state.last_error)
        try:
            frame = self.source.read()
        except Exception as exc:
            self.state.frames_dropped += 1
            self._mark_unavailable(f"capture read failed: {exc}")
            raise
        if frame is None:
            self.state.frames_dropped += 1
            return None
        self.state.available = True
        self.state.capture_wait_ms = frame.capture_wait_ms
        if self._last_frame_ts_ns is not None:
            self.state.frame_period_ms = (frame.ts_ns - self._last_frame_ts_ns) / 1e6
            if self.state.frame_period_ms > 0:
                self.state.fps_capture = 1000.0 / self.state.frame_period_ms
        self._last_frame_ts_ns = frame.ts_ns
        self.state.last_error = None
        return frame

    def capture_frames(
        self,
        *,
        seconds: float | None = None,
        max_frames: int | None = None,
        max_empty_reads: int | None = None,
        max_recoveries: int = 1,
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
        recovery_attempts = 0
        while True:
            if max_frames is not None and count >= max_frames:
                break
            if seconds is not None and time.monotonic() - start >= seconds:
                break
            source = self.source
            if source is None:
                break
            try:
                frame = source.read()
            except Exception as exc:
                self.state.frames_dropped += 1
                read_error = f"capture read failed: {exc}"
                if recovery_attempts < max_recoveries:
                    recovered = self._recover_source()
                    recovery_attempts += 1
                    self.state.recoveries += 1
                    if recovered:
                        self.state.last_error = read_error
                        continue
                    if self.state.last_error:
                        read_error = f"{read_error}; {self.state.last_error}"
                self._mark_unavailable(read_error)
                break
            if frame is None:
                self.state.frames_dropped += 1
                empty_reads += 1
                if empty_read_limit is not None and empty_reads >= empty_read_limit:
                    if recovery_attempts >= max_recoveries:
                        self._mark_unavailable(
                            f"capture produced {empty_read_limit} empty reads"
                        )
                        break
                    recovered = self._recover_source()
                    recovery_attempts += 1
                    self.state.recoveries += 1
                    empty_reads = 0
                    if recovered:
                        continue
                    reason = (
                        self.state.last_error
                        or f"capture recovery failed after {empty_read_limit} empty reads"
                    )
                    self._mark_unavailable(reason)
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

    def _recover_source(self) -> bool:
        if self.state.profile is None:
            return False
        old_source = self.source
        replacement: FrameSource | None = None
        try:
            replacement = self.source_factory(self.state.profile)
        except Exception as exc:
            self.state.last_error = f"capture recovery failed: {exc}"
        close_error = self._close_source(old_source)
        if close_error:
            if replacement is not None:
                replacement_close_error = self._close_source(replacement)
                if replacement_close_error:
                    close_error = f"{close_error}; {replacement_close_error}"
            self.source = None
            self.state.last_error = f"capture recovery failed: {close_error}"
            return False
        self.source = replacement
        if replacement is not None:
            self.state.available = True
            self.state.backend = replacement.backend_label
            self.state.last_error = None
            return True
        return False

    def _mark_unavailable(self, reason: str) -> None:
        source = self.source
        self.source = None
        close_error = self._close_source(source)
        if close_error:
            reason = f"{reason}; {close_error}"
        self.state.available = False
        self.state.last_error = reason
        self._last_frame_ts_ns = None

    def mark_unavailable(self, reason: str) -> None:
        self._mark_unavailable(reason)

    def _close_source(self, source: FrameSource | None) -> str:
        if source is None:
            return ""
        try:
            source.close()
        except Exception as exc:
            return f"capture close failed: {exc}"
        return ""
