from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from novasight.config.runtime import CaptureConfig

from .caps import query_capabilities, run_v4l2_ctl
from .pipeline import CaptureCandidate, build_appsink_candidates, build_pipeline_candidates
from .profile import select_capture_profile
from .session import CaptureSession
from .source import CapturedFrame, FrameSource, GstAppSinkFrameSource, OpenCvFrameSource
from .state import CaptureCapabilities, CaptureProfile, CaptureRuntimeState

logger = logging.getLogger("novasight.capture.service")


def _open_first_readable_source(
    profile: CaptureProfile,
    *,
    candidates: list[CaptureCandidate],
    source_cls: Any,
) -> FrameSource:
    failures: list[str] = []
    for candidate in candidates:
        source: FrameSource | None = None
        try:
            logger.info(
                "capture opening candidate label=%s device=%s",
                candidate.label,
                profile.device,
            )
            source = source_cls(profile, candidate)
            if source.opened_and_readable():
                logger.info(
                    "capture candidate opened label=%s device=%s",
                    candidate.label,
                    profile.device,
                )
                opened_source = source
                source = None
                return opened_source
            failures.append(candidate.label)
        except Exception as exc:
            logger.warning(
                "capture candidate failed label=%s device=%s error=%s",
                candidate.label,
                profile.device,
                exc,
            )
            failures.append(f"{candidate.label}: {exc}")
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    failures.append(f"{candidate.label}: close failed: {exc}")
    raise RuntimeError(
        f"no capture backend opened for {profile.device}: {', '.join(failures)}"
    )


def _open_default_source(profile: CaptureProfile) -> FrameSource:
    appsink_failures: list[str] = []
    try:
        return _open_first_readable_source(
            profile,
            candidates=build_appsink_candidates(profile),
            source_cls=GstAppSinkFrameSource,
        )
    except Exception as exc:
        appsink_failures.append(str(exc))

    try:
        return _open_first_readable_source(
            profile,
            candidates=build_pipeline_candidates(profile),
            source_cls=OpenCvFrameSource,
        )
    except Exception as exc:
        details = "; ".join(appsink_failures + [str(exc)])
        raise RuntimeError(details) from exc


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
        self.session = CaptureSession(
            source_factory=self.source_factory,
            empty_read_sleep_s=empty_read_sleep_s,
        )
        self.state = CaptureRuntimeState(device=config.device)
        self.session.state = self.state
        self.last_config_error: CaptureRuntimeState | None = None
        self._last_preview_output_ts_ns: int | None = None
        self._source_lock = threading.RLock()

    @property
    def source(self) -> FrameSource | None:
        return self.session.source

    def _sync_state(self) -> CaptureRuntimeState:
        self.state = self.session.state
        return self.state

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
        with self._source_lock:
            return self._configure_unlocked(
                device,
                preference=preference,
                pixel_format=pixel_format,
                width=width,
                height=height,
                fps=fps,
            )

    def _configure_unlocked(
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
            self.session.stop(failure.last_error)
            self.session.state = failure
            self.state = failure
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
            self.session.stop(failure.last_error)
            self.session.state = failure
            self.state = failure
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
            self.session.reconfigure(profile)
        except Exception as exc:
            logger.warning(
                "capture configure failed device=%s error=%s",
                selected_device,
                exc,
            )
            failure = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error=str(exc),
            )
            self.last_config_error = failure
            self.session.stop(str(exc))
            self.session.state = failure
            self.state = failure
            return self.state
        self.config.device = selected_device
        self.config.preference = selected_preference
        self.config.pixel_format = selected_pixel_format
        self.config.width = selected_width
        self.config.height = selected_height
        self.config.fps = selected_fps
        self.last_config_error = None
        self._last_preview_output_ts_ns = None
        return self._sync_state()

    def stop(self, reason: str | None = None) -> CaptureRuntimeState:
        with self._source_lock:
            state = self.session.stop(reason)
            return self._sync_state()

    def read_frame(self) -> CapturedFrame | None:
        if self.source is None:
            self.state.last_error = "capture source is not configured"
            raise RuntimeError(self.state.last_error)
        return self.session.latest_frame(timeout_s=0.0)

    def get_latest_preview_frame(self) -> CapturedFrame | None:
        return self.session.latest_frame(timeout_s=0.0)

    def wait_preview_frame(
        self,
        *,
        after_frame_id: int | None = None,
        timeout_s: float = 0.0,
    ) -> CapturedFrame | None:
        return self.session.latest_frame(
            after_frame_id=after_frame_id,
            timeout_s=timeout_s,
        )

    def record_preview_output(self, frame: CapturedFrame, *, target_fps: int) -> None:
        del frame
        with self._source_lock:
            self._sync_state()
            now_ns = time.monotonic_ns()
            self.state.preview_target_fps = target_fps
            self.state.preview_output_frames += 1
            if self._last_preview_output_ts_ns is not None:
                period_ms = (now_ns - self._last_preview_output_ts_ns) / 1e6
                if period_ms > 0:
                    self.state.preview_fps = 1000.0 / period_ms
            self._last_preview_output_ts_ns = now_ns

    def record_preview_drop(self, *, target_fps: int) -> None:
        with self._source_lock:
            self._sync_state()
            self.state.preview_target_fps = target_fps
            self.state.preview_dropped += 1

    def capture_frames(
        self,
        *,
        seconds: float | None = None,
        max_frames: int | None = None,
        max_empty_reads: int | None = None,
        max_recoveries: int = 1,
    ) -> CaptureRuntimeState:
        del max_recoveries
        if self.source is None:
            self._sync_state()
            if self.state.last_error:
                return self.state
            raise RuntimeError("capture source is not configured")
        start = time.monotonic()
        count = 0
        empty_read_limit = max_empty_reads
        if empty_read_limit is None and seconds is None and max_frames is not None:
            empty_read_limit = 100
        initial_drops = self.state.frames_dropped
        last_frame_id: int | None = None
        while True:
            if max_frames is not None and count >= max_frames:
                break
            if seconds is not None and time.monotonic() - start >= seconds:
                break
            frame = self.session.latest_frame(
                after_frame_id=last_frame_id,
                timeout_s=self.empty_read_sleep_s,
            )
            self._sync_state()
            if self.source is None:
                break
            if empty_read_limit is not None:
                empty_reads = self.state.frames_dropped - initial_drops
                if empty_reads >= empty_read_limit:
                    self.stop(f"capture produced {empty_read_limit} empty reads")
                    break
            if frame is None:
                continue
            count += 1
            last_frame_id = frame.frame_id
            self.state.preview_frames += 1
        elapsed = max(time.monotonic() - start, 0.000001)
        if count:
            self.state.fps_capture = count / elapsed
        return self._sync_state()

    def _mark_unavailable(self, reason: str) -> None:
        self.stop(reason)

    def mark_unavailable(self, reason: str) -> None:
        self._mark_unavailable(reason)
