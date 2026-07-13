from __future__ import annotations

from collections import deque
from math import exp, isfinite
from statistics import median
from typing import Deque

from .models import PositionSample, VelocityConfig, VelocityEstimate


class RobustVelocityEstimator:
    """Four-position, three-segment robust X velocity estimator.

    Time is expressed in milliseconds and velocity in pixels per millisecond.
    State belongs to exactly one target ID.
    """

    def __init__(self, config: VelocityConfig) -> None:
        if config.history_size != 4 or config.velocity_sample_count != 3:
            raise ValueError("robust velocity requires four positions and three segments")
        if config.smoothing_tau_ms <= 0.0:
            raise ValueError("smoothing_tau_ms must be > 0")
        if config.history_reset_gap_ms <= 0.0:
            raise ValueError("history_reset_gap_ms must be > 0")
        if config.spread_base_px_ms <= 0.0 or config.change_base_px_ms <= 0.0:
            raise ValueError("velocity confidence base scales must be > 0")
        if config.spread_relative < 0.0 or config.change_relative < 0.0:
            raise ValueError("velocity confidence relative scales must be >= 0")
        self.config = config
        self.target_id: int | None = None
        self.samples: Deque[PositionSample] = deque(maxlen=4)
        self.filtered_velocity = 0.0
        self.initialized_velocity = False
        self.complete_window_updates = 0

    @property
    def history_position_count(self) -> int:
        return len(self.samples)

    def reset(self, target_id: int | None = None) -> None:
        self.target_id = target_id
        self.samples.clear()
        self.filtered_velocity = 0.0
        self.initialized_velocity = False
        self.complete_window_updates = 0

    def update(
        self,
        *,
        target_id: int,
        aim_x: float,
        capture_ts_ns: int,
        detection_confidence: float,
        track_confidence: float = 1.0,
    ) -> VelocityEstimate | None:
        if not isfinite(aim_x) or capture_ts_ns <= 0:
            return None
        if self.target_id != target_id:
            self.reset(target_id)

        if self.samples:
            dt_ms = (capture_ts_ns - self.samples[-1].capture_ts_ns) / 1_000_000.0
            if dt_ms <= 0.0:
                self.reset(target_id)
                return None
            if dt_ms > self.config.history_reset_gap_ms:
                self.reset(target_id)

        self.samples.append(PositionSample(aim_x=aim_x, capture_ts_ns=capture_ts_ns))
        if len(self.samples) < 4:
            return None

        points = tuple(self.samples)
        velocities: list[float] = []
        intervals_ms: list[float] = []
        for previous, current in zip(points, points[1:]):
            dt_ms = (current.capture_ts_ns - previous.capture_ts_ns) / 1_000_000.0
            if dt_ms <= 0.0:
                self.reset(target_id)
                return None
            velocities.append((current.aim_x - previous.aim_x) / dt_ms)
            intervals_ms.append(dt_ms)

        v1, v2, v3 = velocities
        median_velocity = float(median(velocities))
        spread = float(median(abs(value - median_velocity) for value in velocities))
        latest_dt_ms = intervals_ms[-1]

        if not self.initialized_velocity:
            previous_filtered = median_velocity
            self.filtered_velocity = median_velocity
            self.initialized_velocity = True
        else:
            previous_filtered = self.filtered_velocity
            alpha = 1.0 - exp(-latest_dt_ms / self.config.smoothing_tau_ms)
            self.filtered_velocity = (
                previous_filtered * (1.0 - alpha) + median_velocity * alpha
            )

        self.complete_window_updates += 1
        history_quality = min(1.0, self.complete_window_updates / 2.0)
        spread_scale = (
            self.config.spread_base_px_ms
            + self.config.spread_relative * abs(median_velocity)
        )
        spread_quality = 1.0 / (1.0 + spread / max(spread_scale, 1e-9))
        trend_delta = abs(median_velocity - previous_filtered)
        trend_scale = (
            self.config.change_base_px_ms
            + self.config.change_relative * abs(previous_filtered)
        )
        trend_quality = 1.0 / (1.0 + trend_delta / max(trend_scale, 1e-9))
        detection_quality = _clamp(detection_confidence, 0.0, 1.0)
        track_quality = _clamp(track_confidence, 0.0, 1.0)
        motion_confidence = _clamp(
            history_quality
            * spread_quality
            * trend_quality
            * detection_quality
            * track_quality,
            0.0,
            1.0,
        )

        return VelocityEstimate(
            raw_velocities=(v1, v2, v3),
            median_velocity=median_velocity,
            filtered_velocity=self.filtered_velocity,
            spread=spread,
            motion_confidence=motion_confidence,
            measurement_dt_ms=latest_dt_ms,
            history_position_count=len(self.samples),
            history_quality=history_quality,
            spread_quality=spread_quality,
            trend_quality=trend_quality,
            detection_quality=detection_quality,
            track_quality=track_quality,
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
