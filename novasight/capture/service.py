from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from novasight.config.runtime import CaptureConfig

from .caps import query_capabilities, run_v4l2_ctl
from .pipeline import CaptureCandidate, build_appsink_candidates
from .profile import select_capture_profile
from .session import CaptureSession
from .source import CapturedFrame, FrameSource, GstAppSinkFrameSource, ImageFrameSource
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


def _open_default_source(
    profile: CaptureProfile,
    *,
    roi_size: int | None = None,
    roi_offset_x: int = 0,
    roi_offset_y: int = 0,
) -> FrameSource:
    return _open_first_readable_source(
        profile,
        candidates=build_appsink_candidates(
            profile,
            roi_size=roi_size,
            roi_offset_x=roi_offset_x,
            roi_offset_y=roi_offset_y,
        ),
        source_cls=GstAppSinkFrameSource,
    )


class CaptureService:
    def __init__(
        self,
        config: CaptureConfig,
        *,
        capability_runner: Callable[[str], str | None] | None = None,
        source_factory: Callable[[CaptureProfile], FrameSource] | None = None,
        empty_read_sleep_s: float = 0.001,
        roi_size: int | None = None,
        roi_offset_x: int = 0,
        roi_offset_y: int = 0,
    ) -> None:
        self.config = config
        self.roi_size = roi_size
        self.roi_offset_x = int(roi_offset_x)
        self.roi_offset_y = int(roi_offset_y)
        self.capability_runner = capability_runner
        self._default_source_factory = source_factory or self._open_configured_source
        self.source_factory = self._default_source_factory
        self.empty_read_sleep_s = empty_read_sleep_s
        self.session = CaptureSession(
            source_factory=self.source_factory,
            empty_read_sleep_s=empty_read_sleep_s,
        )
        self.state = CaptureRuntimeState(device=config.device)
        self.last_config_error: CaptureRuntimeState | None = None
        self._last_preview_output_ts_ns: int | None = None
        self._preview_window_ts_ns: deque[int] = deque()
        self._source_lock = threading.RLock()

    def _open_configured_source(self, profile: CaptureProfile) -> FrameSource:
        return _open_default_source(
            profile,
            roi_size=self.roi_size,
            roi_offset_x=self.roi_offset_x,
            roi_offset_y=self.roi_offset_y,
        )

    @property
    def state(self) -> CaptureRuntimeState:
        return self.session.state

    @state.setter
    def state(self, value: CaptureRuntimeState) -> None:
        self.session.state = value

    @property
    def source(self) -> FrameSource | None:
        return self.session.source

    def _sync_state(self) -> CaptureRuntimeState:
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
        self.source_factory = self._default_source_factory
        self.session.source_factory = self._default_source_factory
        selected_device = self.config.device if device is None else device
        if not selected_device.strip():
            failure = CaptureRuntimeState(
                available=False,
                device=selected_device,
                last_error="capture device is required",
            )
            self.last_config_error = failure
            self.session.stop(failure.last_error)
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
            self.state = failure
            return self.state
        self.config.device = profile.device
        self.config.preference = "manual"
        self.config.pixel_format = profile.pixel_format
        self.config.width = profile.width
        self.config.height = profile.height
        self.config.fps = profile.fps
        self.last_config_error = None
        self._last_preview_output_ts_ns = None
        self._preview_window_ts_ns.clear()
        return self._sync_state()

    def configure_image(self, path: str, *, fps: int = 30) -> CaptureRuntimeState:
        with self._source_lock:
            image_source = ImageFrameSource(path, fps=fps)
            profile = CaptureProfile(
                device=path,
                pixel_format="IMAGE",
                width=image_source.width,
                height=image_source.height,
                fps=image_source.fps,
                preference="image",
                selection_reason="image source",
            )

            def factory(_: CaptureProfile) -> FrameSource:
                return image_source

            previous_factory = self.session.source_factory
            self.session.source_factory = factory
            try:
                self.session.reconfigure(profile)
            except Exception:
                self.session.source_factory = previous_factory
                self.source_factory = previous_factory
                image_source.close()
                raise
            self.source_factory = factory
            self.config.device = path
            self.config.preference = "image"
            self.config.pixel_format = "IMAGE"
            self.config.width = image_source.width
            self.config.height = image_source.height
            self.config.fps = image_source.fps
            self.last_config_error = None
            self._last_preview_output_ts_ns = None
            self._preview_window_ts_ns.clear()
            return self._sync_state()

    def stop(self, reason: str | None = None) -> CaptureRuntimeState:
        with self._source_lock:
            self.session.stop(reason)
            return self._sync_state()

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
        with self.session._condition:
            now_ns = time.monotonic_ns()
            self.state.preview_target_fps = target_fps
            self.state.preview_output_frames += 1
            self._preview_window_ts_ns.append(now_ns)
            window_start_ns = now_ns - 1_000_000_000
            while self._preview_window_ts_ns and self._preview_window_ts_ns[0] < window_start_ns:
                self._preview_window_ts_ns.popleft()
            self.state.preview_fps = self._window_fps(self._preview_window_ts_ns)
            self._last_preview_output_ts_ns = now_ns

    def record_preview_drop(self, *, target_fps: int) -> None:
        with self.session._condition:
            self.state.preview_target_fps = target_fps
            self.state.preview_dropped += 1

    def _window_fps(self, timestamps_ns: deque[int]) -> float:
        if len(timestamps_ns) < 2:
            return 0.0
        elapsed_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        if elapsed_s <= 0:
            return 0.0
        return (len(timestamps_ns) - 1) / elapsed_s
