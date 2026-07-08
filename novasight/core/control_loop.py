from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Protocol

from novasight.contracts import ControlIntent, FrameContext, Track


@dataclass(frozen=True)
class ControlLoopCalibration:
    counts_per_360_x: float
    counts_per_360_y: float


@dataclass
class ControlLoopStats:
    processed_frames: int = 0
    skipped_frames: int = 0
    cancelled_frames: int = 0
    emitted_intents: int = 0
    last_frame_id: int = -1
    last_error: str = ""
    last_intent: ControlIntent | None = None
    last_target_id: int | None = None
    last_control_ts_ns: int = 0
    last_stage_timestamps: dict[str, int] = field(default_factory=dict)


class FrameContextSlot(Protocol):
    version: int

    def get(self, *, after_version: int = 0, timeout_s: float = 0.0) -> FrameContext | None:
        ...


class TargetSelectorProtocol(Protocol):
    def select(self, tracks: list[Track], now_ns: int) -> Track | None:
        ...


class KalmanProtocol(Protocol):
    def update(self, track: Track, now_ns: int) -> None:
        ...

    def predict(self, now_ns: int) -> Track | None:
        ...


class LatencyCompensatorProtocol(Protocol):
    def compute_horizon(self, measurement_ns: int, now_ns: int) -> int:
        ...


class AngularControllerProtocol(Protocol):
    def compute(self, error_px: tuple[float, float], now_ns: int) -> tuple[float, float, float, float]:
        ...


class ControlLoop(threading.Thread):
    def __init__(
        self,
        *,
        slot: FrameContextSlot,
        selector: TargetSelectorProtocol,
        kalman: KalmanProtocol | None,
        latency_compensator: LatencyCompensatorProtocol,
        controller: AngularControllerProtocol,
        intent_sink: Any,
        calibration: ControlLoopCalibration,
        min_interval_s: float = 0.005,
        poll_timeout_s: float = 0.05,
        confidence: float = 1.0,
        name: str = "novasight-control-loop",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.slot = slot
        self.selector = selector
        self.kalman = kalman
        self.latency_compensator = latency_compensator
        self.controller = controller
        self.intent_sink = intent_sink
        self.calibration = calibration
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.poll_timeout_s = max(0.0, float(poll_timeout_s))
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.stats = ControlLoopStats()
        self._stop_event = threading.Event()
        self._last_slot_version = 0

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=max(0.0, float(timeout_s)))

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                started_s = time.monotonic()
                context = self.slot.get(
                    after_version=self._last_slot_version,
                    timeout_s=self.poll_timeout_s,
                )
                if context is None:
                    continue
                current_version = int(getattr(self.slot, "version", self._last_slot_version + 1))
                self.stats.skipped_frames += max(0, current_version - self._last_slot_version - 1)
                self._last_slot_version = current_version
                self._process_context(context, now_ns=time.monotonic_ns())
                self._sleep_for_pacing(started_s)
        except Exception as exc:
            self.stats.last_error = str(exc)

    def _process_context(self, context: FrameContext, *, now_ns: int) -> None:
        self.stats.processed_frames += 1
        self.stats.last_frame_id = int(context.frame_id)
        self.stats.last_stage_timestamps = _stage_timestamps(context, control_start_ts_ns=now_ns)
        if not context.tracks:
            self._cancel_pending("no_tracks")
            return

        target = self.selector.select(list(context.tracks), now_ns)
        if target is None:
            self._cancel_pending("no_target")
            return
        if bool(getattr(target, "is_predicted", False)) and self.kalman is not None:
            predicted = self.kalman.predict(now_ns)
            if predicted is None:
                self._cancel_pending("prediction_unavailable")
                return
            target = predicted
        elif self.kalman is not None:
            self.kalman.update(target, now_ns)

        intent = self._intent_for_target(context, target, now_ns=now_ns)
        self._dispatch_intent(intent, now_ns)
        self.stats.emitted_intents += 1
        self.stats.last_intent = intent
        self.stats.last_target_id = int(target.track_id)
        self.stats.last_control_ts_ns = now_ns

    def _intent_for_target(self, context: FrameContext, target: Track, *, now_ns: int) -> ControlIntent:
        last_seen_ns = _last_seen_ns(context, target)
        horizon_ns = self.latency_compensator.compute_horizon(last_seen_ns, now_ns)
        velocity_x, velocity_y = _velocity_px_s(target)
        predicted_center = (
            float(target.cx) + velocity_x * horizon_ns * 1e-9,
            float(target.cy) + velocity_y * horizon_ns * 1e-9,
        )
        _ex_rad, _ey_rad, control_x_rad, control_y_rad = self.controller.compute(
            predicted_center,
            now_ns,
        )
        counts_x = control_x_rad * float(self.calibration.counts_per_360_x) / math.tau
        counts_y = control_y_rad * float(self.calibration.counts_per_360_y) / math.tau
        return ControlIntent(
            dx=counts_x,
            dy=counts_y,
            action="move",
            confidence=self.confidence,
            reason="control_loop",
            source_id="control_loop",
            source_frame_id=int(context.frame_id),
            source_track_id=int(target.track_id),
            predicted_source=bool(getattr(target, "is_predicted", False)),
        )

    def _dispatch_intent(self, intent: ControlIntent, now_ns: int) -> None:
        schedule = getattr(self.intent_sink, "schedule", None)
        if callable(schedule):
            schedule(intent, now_ns)
            return
        execute = getattr(self.intent_sink, "execute", None)
        if callable(execute):
            execute(intent)
            return
        submit = getattr(self.intent_sink, "submit", None)
        if callable(submit):
            submit(intent)
            return
        raise TypeError("intent_sink must expose schedule(intent, now_ns), execute(intent), or submit(intent)")

    def _cancel_pending(self, reason: str) -> None:
        self.stats.cancelled_frames += 1
        cancel = getattr(self.intent_sink, "cancel_all", None)
        if callable(cancel):
            cancel()
            return
        clear = getattr(self.intent_sink, "clear", None)
        if callable(clear):
            clear(reason)

    def _sleep_for_pacing(self, started_s: float) -> None:
        remaining_s = self.min_interval_s - (time.monotonic() - started_s)
        if remaining_s > 0.0:
            self._stop_event.wait(remaining_s)


def _velocity_px_s(target: Track) -> tuple[float, float]:
    value = getattr(target, "velocity_px_s", (0.0, 0.0))
    if not isinstance(value, tuple) or len(value) != 2:
        return 0.0, 0.0
    return float(value[0]), float(value[1])


def _last_seen_ns(context: FrameContext, target: Track) -> int:
    value = getattr(target, "last_seen_ns", None)
    if value is not None:
        return int(value)
    for candidate in (
        context.postprocess_ts_ns,
        context.inference_end_ts_ns,
        context.capture_ts_ns,
    ):
        if candidate is not None:
            return int(candidate)
    return 0


def _stage_timestamps(context: FrameContext, *, control_start_ts_ns: int) -> dict[str, int]:
    values = {
        "capture_ts_ns": context.capture_ts_ns,
        "dequeue_ts_ns": context.dequeue_ts_ns,
        "decode_ts_ns": context.decode_ts_ns,
        "roi_ts_ns": context.roi_ts_ns,
        "inference_start_ts_ns": context.inference_start_ts_ns,
        "inference_end_ts_ns": context.inference_end_ts_ns,
        "postprocess_ts_ns": context.postprocess_ts_ns,
        "control_start_ts_ns": control_start_ts_ns,
    }
    return {key: int(value) for key, value in values.items() if value is not None}


__all__ = [
    "ControlLoop",
    "ControlLoopCalibration",
    "ControlLoopStats",
]
