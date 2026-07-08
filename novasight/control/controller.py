from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class AngularControllerCalibration:
    width: float
    height: float
    fov_x: float
    fov_y: float


@dataclass(frozen=True)
class AngularControllerConfig:
    kp: float = 1.0
    kd: float = 0.0


class AngularController:
    def __init__(
        self,
        calib: AngularControllerCalibration | Any,
        config: AngularControllerConfig | Any,
    ) -> None:
        self.width = max(1.0, float(getattr(calib, "width")))
        self.height = max(1.0, float(getattr(calib, "height")))
        self.fov_x = _radians(getattr(calib, "fov_x"))
        self.fov_y = _radians(getattr(calib, "fov_y"))
        self.focal_x = self.width / (2.0 * math.tan(self.fov_x / 2.0))
        self.focal_y = self.height / (2.0 * math.tan(self.fov_y / 2.0))
        self.kp = float(getattr(config, "kp"))
        self.kd = float(getattr(config, "kd"))
        self.prev_error_rad = (0.0, 0.0)
        self.prev_time_ns = 0

    def compute(self, error_px: tuple[float, float], now_ns: int) -> tuple[float, float, float, float]:
        ex = float(error_px[0]) - (self.width / 2.0)
        ey = float(error_px[1]) - (self.height / 2.0)
        ex_rad = math.atan(ex / self.focal_x)
        ey_rad = math.atan(ey / self.focal_y)
        dt = (int(now_ns) - int(self.prev_time_ns)) * 1e-9
        if dt <= 0.0 or not math.isfinite(dt):
            dx_dt = 0.0
            dy_dt = 0.0
        else:
            dx_dt = (ex_rad - self.prev_error_rad[0]) / dt
            dy_dt = (ey_rad - self.prev_error_rad[1]) / dt
        control_x_rad = self.kp * ex_rad + self.kd * dx_dt
        control_y_rad = self.kp * ey_rad + self.kd * dy_dt
        self.prev_error_rad = (ex_rad, ey_rad)
        self.prev_time_ns = int(now_ns)
        return ex_rad, ey_rad, control_x_rad, control_y_rad


def _radians(value: Any) -> float:
    number = float(value)
    if number > math.tau:
        number = math.radians(number)
    if not 0.0 < number < math.pi:
        raise ValueError("field of view must be in (0, pi) radians or (0, 180) degrees")
    return number


__all__ = [
    "AngularController",
    "AngularControllerCalibration",
    "AngularControllerConfig",
]
