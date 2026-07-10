from __future__ import annotations

from dataclasses import dataclass


NS_PER_SECOND = 1_000_000_000
MS_PER_SECOND = 1_000.0


@dataclass(frozen=True, slots=True)
class ControlTimingSnapshot:
    frame_id: int
    target_id: int | None
    capture_ts_ns: int
    inference_end_ts_ns: int | None
    control_now_ts_ns: int
    measurement_dt_s: float | None
    frame_age_s: float
    configured_actuation_delay_s: float
    prediction_horizon_s: float

    def as_telemetry(self) -> dict[str, int | float | str | None]:
        return {
            "frame_id": self.frame_id,
            "target_id": self.target_id,
            "capture_ts_ns": self.capture_ts_ns,
            "inference_end_ts_ns": self.inference_end_ts_ns,
            "control_now_ts_ns": self.control_now_ts_ns,
            "measurement_dt_ms": (
                self.measurement_dt_s * MS_PER_SECOND
                if self.measurement_dt_s is not None
                else None
            ),
            "frame_age_ms": self.frame_age_s * MS_PER_SECOND,
            "configured_actuation_delay_s": self.configured_actuation_delay_s,
            "actuation_delay_source": "configured_estimate",
            "prediction_horizon_ms": self.prediction_horizon_s * MS_PER_SECOND,
        }


class ControlTimingModel:
    """Maintains capture-clock timing state for consecutive target observations."""

    def __init__(self) -> None:
        self._previous_target_id: int | None = None
        self._previous_capture_ts_ns: int | None = None

    def observe(
        self,
        *,
        frame_id: int,
        target_id: int | None,
        capture_ts_ns: int,
        inference_end_ts_ns: int | None,
        control_now_ts_ns: int,
        configured_actuation_delay_s: float,
    ) -> ControlTimingSnapshot:
        capture_ns = int(capture_ts_ns)
        control_now_ns = int(control_now_ts_ns)
        if capture_ns <= 0:
            raise ValueError("capture_ts_ns must be positive")
        if control_now_ns <= 0:
            raise ValueError("control_now_ts_ns must be positive")

        normalized_target_id = int(target_id) if target_id is not None else None
        measurement_dt_s = self._measurement_dt_s(
            target_id=normalized_target_id,
            capture_ts_ns=capture_ns,
        )
        self._advance_measurement_cursor(
            target_id=normalized_target_id,
            capture_ts_ns=capture_ns,
        )

        frame_age_s = max(0.0, (control_now_ns - capture_ns) / NS_PER_SECOND)
        actuation_delay_s = max(0.0, float(configured_actuation_delay_s))
        return ControlTimingSnapshot(
            frame_id=int(frame_id),
            target_id=normalized_target_id,
            capture_ts_ns=capture_ns,
            inference_end_ts_ns=(
                int(inference_end_ts_ns) if inference_end_ts_ns is not None else None
            ),
            control_now_ts_ns=control_now_ns,
            measurement_dt_s=measurement_dt_s,
            frame_age_s=frame_age_s,
            configured_actuation_delay_s=actuation_delay_s,
            prediction_horizon_s=frame_age_s + actuation_delay_s,
        )

    def reset(self) -> None:
        self._previous_target_id = None
        self._previous_capture_ts_ns = None

    def _measurement_dt_s(self, *, target_id: int | None, capture_ts_ns: int) -> float | None:
        if target_id is None or target_id != self._previous_target_id:
            return None
        if self._previous_capture_ts_ns is None or capture_ts_ns <= self._previous_capture_ts_ns:
            return None
        return (capture_ts_ns - self._previous_capture_ts_ns) / NS_PER_SECOND

    def _advance_measurement_cursor(self, *, target_id: int | None, capture_ts_ns: int) -> None:
        if target_id is None:
            self.reset()
            return
        if target_id != self._previous_target_id:
            self._previous_target_id = target_id
            self._previous_capture_ts_ns = capture_ts_ns
            return
        if self._previous_capture_ts_ns is None or capture_ts_ns > self._previous_capture_ts_ns:
            self._previous_capture_ts_ns = capture_ts_ns


__all__ = ["ControlTimingModel", "ControlTimingSnapshot"]
