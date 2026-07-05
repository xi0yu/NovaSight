from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from novasight.contracts import Detection, Track
from novasight.hardware import BoxInputState

from .strategy import MOVE_KINDS, MoveCommand


Target = Detection | Track


@dataclass(slots=True)
class DynamicPidAxis:
    proportional_gain: float
    integral_gain: float
    derivative_gain: float
    target_error_threshold: float = 0.016
    speed_multiplier: float = 1.0
    min_coefficient: float = 1.6
    max_coefficient: float = 2.7
    transition_sharpness: float = 5.0
    dynamic_transition_midpoint: float = 0.0
    minimum_data_count: int = 2
    error_change_tolerance: float = 0.012
    smoothing_factor: float = 1.0

    total_output: float = 0.0
    previous_error: float = 0.0
    previous_smooth_output: float = 0.0
    integral_accumulator: float = 0.0
    loaded_frames: int = 0
    current_speed: float = 0.0
    error_rate: float = 0.0
    previous_frame_speed: float = 0.0
    target_reached: bool = False
    dynamic_judgement_threshold: float = 0.5
    stable_count: int = 0
    proportional: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    derivative_output: float = 0.0
    new_speed: float = 0.0

    def control_loop(
        self,
        current_error: float,
        delta_time: float,
        recent_target_extent: float,
        image_size: float,
    ) -> float:
        self.loaded_frames += 1
        dt = max(1e-6, float(delta_time))
        width_ratio = float(recent_target_extent) / max(1e-6, float(image_size))
        dynamic_coefficient = self.min_coefficient + (
            self.max_coefficient - self.min_coefficient
        ) / (
            1.0 + math.exp(-self.transition_sharpness * (width_ratio - self.dynamic_transition_midpoint))
        )
        self.dynamic_judgement_threshold = dynamic_coefficient * float(recent_target_extent)

        if not self.target_reached and abs(current_error) < self.target_error_threshold:
            self.target_reached = True
        elif abs(current_error) >= self.dynamic_judgement_threshold:
            self.target_reached = False
            self.integral_accumulator = 0.0
            self.stable_count = 0
        elif (
            not self.target_reached
            and abs(current_error) >= self.target_error_threshold
            and abs(current_error) <= self.dynamic_judgement_threshold
        ):
            difference = abs(current_error - self.previous_error)
            if difference < self.error_change_tolerance:
                self.stable_count += 1
            else:
                self.stable_count = 0
            if self.stable_count >= self.minimum_data_count:
                self.target_reached = True
                self.stable_count = 0
                self.integral_accumulator = 0.0

        self.error_rate = (current_error - self.previous_error) / dt
        if self.target_reached:
            self.integral_accumulator += current_error * dt
            self.integral = self.integral_gain * self.integral_accumulator
            self.proportional = self.proportional_gain * current_error
            self.derivative = self.derivative_gain * self.error_rate
            self.derivative_output = self.derivative
        else:
            self.integral_accumulator += (current_error * 0.5) * dt
            self.integral = self.integral_gain * self.integral_accumulator
            self.proportional = (self.proportional_gain * 0.5) * current_error
            self.derivative = self.derivative_gain * self.error_rate
            self.derivative_output = self.derivative

        raw_output = self.proportional + self.integral + self.derivative_output
        self.total_output = self.smoothing_factor * raw_output
        self.previous_smooth_output = self.total_output
        if self.target_reached:
            self.new_speed = (
                (current_error - self.previous_error) / dt
                + (raw_output / dt) * self.speed_multiplier
            )
        else:
            self.new_speed = 0.0

        self.current_speed = self.new_speed
        self.previous_frame_speed = self.new_speed
        self.previous_error = current_error
        return self.total_output

    def update_parameters(
        self,
        proportional_gain: float,
        integral_gain: float,
        derivative_gain: float,
    ) -> None:
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.derivative_gain = derivative_gain

    def get_speed(self) -> float:
        return self.current_speed

    def reset(self) -> None:
        self.total_output = 0.0
        self.previous_error = 0.0
        self.previous_smooth_output = 0.0
        self.integral_accumulator = 0.0
        self.loaded_frames = 0
        self.current_speed = 0.0
        self.error_rate = 0.0
        self.previous_frame_speed = 0.0
        self.target_reached = False
        self.dynamic_judgement_threshold = 0.0
        self.stable_count = 0

    def set_low_level_parameters(
        self,
        target_error_threshold: float,
        speed_multiplier: float,
        min_coefficient: float,
        max_coefficient: float,
        transition_sharpness: float,
        dynamic_transition_midpoint: float,
        minimum_data_count: float,
        error_change_tolerance: float,
    ) -> None:
        self.target_error_threshold = target_error_threshold
        self.speed_multiplier = speed_multiplier
        self.min_coefficient = min_coefficient
        self.max_coefficient = max_coefficient
        self.transition_sharpness = transition_sharpness
        self.dynamic_transition_midpoint = dynamic_transition_midpoint
        self.minimum_data_count = minimum_data_count
        self.error_change_tolerance = error_change_tolerance

    def set_smoothing_factor(self, alpha: float) -> None:
        self.smoothing_factor = max(0.0, min(1.0, float(alpha)))

    def debug(self) -> dict[str, Any]:
        return {
            "target_reached": self.target_reached,
            "dynamic_judgement_threshold": self.dynamic_judgement_threshold,
            "stable_count": self.stable_count,
            "proportional": self.proportional,
            "integral": self.integral,
            "derivative": self.derivative,
            "derivative_output": self.derivative_output,
            "integral_accumulator": self.integral_accumulator,
            "error_rate": self.error_rate,
            "current_speed": self.current_speed,
            "loaded_frames": self.loaded_frames,
            "total_output": self.total_output,
        }


@dataclass(frozen=True, slots=True)
class DynamicPidConfig:
    kp_x: float = 0.35
    kp_y: float = 0.24
    ki: float = 0.0
    kd: float = 0.0
    target_error_threshold: float = 0.016
    speed_multiplier: float = 1.0
    min_coefficient: float = 1.6
    max_coefficient: float = 2.7
    transition_sharpness: float = 5.0
    dynamic_transition_midpoint: float = 0.0
    minimum_data_count: int = 2
    error_change_tolerance: float = 0.012
    smoothing_factor: float = 1.0
    aim_ratio: float = 40.0
    fov_deg: float = 105.0
    counts_per_revolution_x: float = 9980.0
    counts_per_revolution_y: float = 9980.0
    control_hz: float = 60.0
    ema_alpha: float = 0.45
    move_kind: str = "raw"
    move_ms: int = 0
    trace_ms: int = 0


class DynamicPidMouseStrategy:
    def __init__(self, config: DynamicPidConfig | None = None) -> None:
        self.config = config or DynamicPidConfig()
        self.axis_x = _axis_from_config(self.config, axis="x")
        self.axis_y = _axis_from_config(self.config, axis="y")
        self._last_s = 0.0
        self._last_capture_ts_ns: int | None = None
        self._last_filtered_aim: tuple[float, float] | None = None

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        raw = box_input.raw or {}
        target_key = str(raw.get("target_key") or "")
        observed_dt, observed_dt_source = self._resolve_delta_time(raw)
        control_hz = max(1.0, float(self.config.control_hz))
        dt = 1.0 / control_hz
        center_x, center_y = current_pos
        raw_aim_x, raw_aim_y = _aim_point(target, self.config.aim_ratio)
        aim_x, aim_y = self._filter_aim(raw_aim_x, raw_aim_y)
        error_x_px = float(aim_x - center_x)
        error_y_px = float(aim_y - center_y)
        fov_deg = min(179.0, max(1.0, float(self.config.fov_deg)))
        fov_rad = math.radians(fov_deg)
        roi_width_px = max(1.0, center_x * 2.0)
        focal_px = max(1e-6, (roi_width_px * 0.5) / math.tan(fov_rad * 0.5))
        error_x_rad = math.atan(error_x_px / focal_px)
        error_y_rad = math.atan(error_y_px / focal_px)
        target_width_rad = max(1e-6, 2.0 * math.atan(max(1.0, float(target.w)) * 0.5 / focal_px))
        pid_output_x_rad = self.axis_x.control_loop(error_x_rad, dt, target_width_rad, fov_rad)
        pid_output_y_rad = self.axis_y.control_loop(error_y_rad, dt, target_width_rad, fov_rad)
        counts_per_revolution_x = max(1.0, float(self.config.counts_per_revolution_x))
        counts_per_revolution_y = max(1.0, float(self.config.counts_per_revolution_y))
        output_x_counts = pid_output_x_rad / (2.0 * math.pi) * counts_per_revolution_x
        output_y_counts = pid_output_y_rad / (2.0 * math.pi) * counts_per_revolution_y
        move_kind = self.config.move_kind if self.config.move_kind in MOVE_KINDS else "raw"
        return MoveCommand(
            dx=output_x_counts,
            dy=output_y_counts,
            confidence=float(getattr(target, "score", 1.0)),
            reason="dynamic pid mouse strategy",
            move_kind=move_kind,
            move_ms=max(0, int(self.config.move_ms)),
            trace_ms=max(0, int(self.config.trace_ms)),
            debug={
                "stage": "dynamic_pid_pipeline",
                "algorithm": "dynamic_pid",
                "unit_pipeline": "bbox_px_to_angle_rad_to_counts",
                "coordinate_y": "image_down_positive_before_executor",
                "aim_ratio": self.config.aim_ratio,
                "raw_aim_x": raw_aim_x,
                "raw_aim_y": raw_aim_y,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "ema_alpha": self.config.ema_alpha,
                "error_x_px": error_x_px,
                "error_y_px": error_y_px,
                "error_x_rad": error_x_rad,
                "error_y_rad": error_y_rad,
                "pid_error_x": error_x_rad,
                "pid_error_y": error_y_rad,
                "recent_target_width_px": max(1.0, float(target.w)),
                "recent_target_width_rad": target_width_rad,
                "fov_deg": fov_deg,
                "fov_rad": fov_rad,
                "focal_px": focal_px,
                "counts_per_revolution_x": counts_per_revolution_x,
                "counts_per_revolution_y": counts_per_revolution_y,
                "control_hz": control_hz,
                "dt_ms": dt * 1000.0,
                "dt_source": "fixed_control_hz",
                "observed_dt_ms": observed_dt * 1000.0,
                "observed_dt_source": observed_dt_source,
                "frame_id": raw.get("frame_id"),
                "capture_ts_ns": raw.get("capture_ts_ns"),
                "target_key": target_key,
                "target_extent": target_width_rad,
                "image_size": fov_rad,
                "pid_output_x_rad": pid_output_x_rad,
                "pid_output_y_rad": pid_output_y_rad,
                "output_x_counts": output_x_counts,
                "output_y_counts": output_y_counts,
                "raw_output_x": output_x_counts,
                "raw_output_y": output_y_counts,
                "x_axis": self.axis_x.debug(),
                "y_axis": self.axis_y.debug(),
                "p_x": self.axis_x.proportional,
                "p_y": self.axis_y.proportional,
                "i_x": self.axis_x.integral,
                "i_y": self.axis_y.integral,
                "d_x": self.axis_x.derivative_output,
                "d_y": self.axis_y.derivative_output,
                "trigger_active": bool(box_input.active),
            },
        )

    def reset(self) -> None:
        self.axis_x.reset()
        self.axis_y.reset()
        self._last_s = 0.0
        self._last_capture_ts_ns = None
        self._last_filtered_aim = None

    def _filter_aim(self, aim_x: float, aim_y: float) -> tuple[float, float]:
        alpha = max(0.0, min(1.0, float(self.config.ema_alpha)))
        previous = self._last_filtered_aim
        if previous is None or alpha <= 0.0:
            filtered = (float(aim_x), float(aim_y))
        else:
            filtered = (
                alpha * float(aim_x) + (1.0 - alpha) * previous[0],
                alpha * float(aim_y) + (1.0 - alpha) * previous[1],
            )
        self._last_filtered_aim = filtered
        return filtered

    def _resolve_delta_time(self, raw: dict[str, object]) -> tuple[float, str]:
        capture_ts_ns = raw.get("capture_ts_ns")
        if isinstance(capture_ts_ns, (int, float)) and capture_ts_ns > 0:
            current_ts_ns = int(capture_ts_ns)
            if self._last_capture_ts_ns is not None:
                delta_ns = current_ts_ns - self._last_capture_ts_ns
                self._last_capture_ts_ns = current_ts_ns
                if delta_ns > 0:
                    return max(1e-6, delta_ns / 1e9), "capture_ts_ns"
            self._last_capture_ts_ns = current_ts_ns
            return 1.0 / 120.0, "capture_ts_ns:first"

        now_s = time.monotonic()
        dt = now_s - self._last_s if self._last_s > 0 else 1.0 / 120.0
        self._last_s = now_s
        return max(1e-6, dt), "monotonic_fallback"


def _axis_from_config(config: DynamicPidConfig, *, axis: str) -> DynamicPidAxis:
    gain = config.kp_x if axis == "x" else config.kp_y
    return DynamicPidAxis(
        proportional_gain=gain,
        integral_gain=config.ki,
        derivative_gain=config.kd,
        target_error_threshold=config.target_error_threshold,
        speed_multiplier=config.speed_multiplier,
        min_coefficient=config.min_coefficient,
        max_coefficient=config.max_coefficient,
        transition_sharpness=config.transition_sharpness,
        dynamic_transition_midpoint=config.dynamic_transition_midpoint,
        minimum_data_count=int(config.minimum_data_count),
        error_change_tolerance=float(config.error_change_tolerance),
        smoothing_factor=config.smoothing_factor,
    )


def _aim_point(target: Target, aim_ratio: float) -> tuple[float, float]:
    ratio = max(0.0, min(100.0, float(aim_ratio))) / 100.0
    return float(target.cx), float(target.y) + float(target.h) * ratio
