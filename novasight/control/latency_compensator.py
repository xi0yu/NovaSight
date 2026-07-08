from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LatencyCalibration:
    actuation_delay_ns: int = 0


class LatencyCompensator:
    def __init__(
        self,
        calibration: LatencyCalibration | Any | None = None,
        *,
        smoothing: float = 0.1,
    ) -> None:
        self.actuation_delay_ns = max(0, int(getattr(calibration, "actuation_delay_ns", 0) or 0))
        self.estimated_output_latency_ns = 0
        self.smoothing = max(0.0, min(1.0, float(smoothing)))

    def compute_horizon(self, measurement_ns: int, now_ns: int) -> int:
        measurement_age = max(0, int(now_ns) - int(measurement_ns))
        return int(
            measurement_age
            + int(self.estimated_output_latency_ns)
            + int(self.actuation_delay_ns)
        )

    def update_latency_estimate(self, total_latency_ns: int) -> int:
        latency_ns = max(0, int(total_latency_ns))
        alpha = float(self.smoothing)
        estimated = (1.0 - alpha) * float(self.estimated_output_latency_ns) + alpha * float(latency_ns)
        self.estimated_output_latency_ns = int(round(estimated))
        return self.estimated_output_latency_ns


__all__ = [
    "LatencyCalibration",
    "LatencyCompensator",
]
