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


def _same_capture_mode(left: CaptureProfile | None, right: CaptureProfile) -> bool:
    if left is None:
        return False
    return (
        left.device == right.device
        and left.pixel_format == right.pixel_format
        and left.width == right.width
        and left.height == right.height
        and left.fps == right.fps
    )


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
        self._last_preview_output_ts_ns: int | None = None
        self._latest_preview_frame: CapturedFrame | None = None
        self._source_lock = threading.RLock()
        self._preview_condition = threading.Condition(self._source_lock)

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
            if self.source is not None and _same_capture_mode(self.state.profile, profile):
                logger.info(
                    "capture configure reused source device=%s format=%s size=%sx%s fps=%s",
                    profile.device,
                    profile.pixel_format,
                    profile.width,
                    profile.height,
                    profile.fps,
                )
                self.config.device = selected_device
                self.config.preference = selected_preference
                self.config.pixel_format = selected_pixel_format
                self.config.width = selected_width
                self.config.height = selected_height
                self.config.fps = selected_fps
                self.last_config_error = None
                self.state.profile = profile
                self.state.available = True
                self.state.last_error = None
                return self.state
            old_source = self.source
            close_before_open = (
                old_source is not None
                and self.state.profile is not None
                and self.state.profile.device == profile.device
            )
            if close_before_open:
                logger.info(
                    "capture closing current source before same-device reconfigure "
                    "device=%s backend=%s",
                    self.state.profile.device,
                    self.state.backend,
                )
                self.source = None
                close_error = self._close_source(old_source)
                if close_error:
                    raise RuntimeError(close_error)
            source = self.source_factory(profile)
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
            if self.source is None:
                self.state = failure
                return self.state
            return self.state
        self.source = source
        if old_source is not None and not close_before_open:
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
        self._last_preview_output_ts_ns = None
        self._latest_preview_frame = None
        self._preview_condition.notify_all()
        return self.state

    def read_frame(self) -> CapturedFrame | None:
        with self._source_lock:
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
            self._publish_preview_frame(frame)
            return frame

    def _publish_preview_frame(self, frame: CapturedFrame) -> None:
        self._latest_preview_frame = frame
        self.state.preview_frames += 1
        self._preview_condition.notify_all()

    def get_latest_preview_frame(self) -> CapturedFrame | None:
        with self._source_lock:
            return self._latest_preview_frame

    def wait_preview_frame(
        self,
        *,
        after_frame_id: int | None = None,
        timeout_s: float = 0.0,
    ) -> CapturedFrame | None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self._preview_condition:
            while True:
                frame = self._latest_preview_frame
                if frame is not None and (
                    after_frame_id is None or frame.frame_id > after_frame_id
                ):
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._preview_condition.wait(remaining)

    def record_preview_output(self, frame: CapturedFrame, *, target_fps: int) -> None:
        del frame
        with self._source_lock:
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
            with self._source_lock:
                self._publish_preview_frame(frame)
        elapsed = max(time.monotonic() - start, 0.000001)
        self.state.fps_capture = count / elapsed
        return self.state

    def _recover_source(self) -> bool:
        if self.state.profile is None:
            return False
        old_source = self.source
        replacement: FrameSource | None = None
        close_error = self._close_source(old_source)
        if close_error:
            self.source = None
            self.state.last_error = f"capture recovery failed: {close_error}"
            return False
        self.source = None
        try:
            replacement = self.source_factory(self.state.profile)
        except Exception as exc:
            self.state.last_error = f"capture recovery failed: {exc}"
        self.source = replacement
        if replacement is not None:
            self.state.available = True
            self.state.backend = replacement.backend_label
            self.state.last_error = None
            return True
        return False

    def _mark_unavailable(self, reason: str) -> None:
        with self._source_lock:
            source = self.source
            self.source = None
            close_error = self._close_source(source)
            if close_error:
                reason = f"{reason}; {close_error}"
            self.state.available = False
            self.state.last_error = reason
            self._last_frame_ts_ns = None
            self._latest_preview_frame = None
            self._preview_condition.notify_all()

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
