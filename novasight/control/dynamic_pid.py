from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
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
    target_error_threshold: float = 4.0
    speed_multiplier: float = 1.0
    min_coefficient: float = 1.6
    max_coefficient: float = 2.7
    transition_sharpness: float = 5.0
    dynamic_transition_midpoint: float = 0.0
    minimum_data_count: int = 2
    error_change_tolerance: int = 3
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
    dynamic_judgement_threshold: float = 0.0
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
    kd: float = 0.1
    target_error_threshold: float = 4.0
    speed_multiplier: float = 1.0
    min_coefficient: float = 1.6
    max_coefficient: float = 2.7
    transition_sharpness: float = 5.0
    dynamic_transition_midpoint: float = 0.0
    minimum_data_count: int = 2
    error_change_tolerance: int = 3
    smoothing_factor: float = 1.0
    aim_ratio: float = 40.0
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
        self._last_target_key = ""

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        raw = box_input.raw or {}
        target_key = str(raw.get("target_key") or "")
        target_changed = bool(target_key and self._last_target_key and target_key != self._last_target_key)
        if target_changed:
            self.axis_x.reset()
            self.axis_y.reset()
            self._last_capture_ts_ns = None
            self._last_s = 0.0
        if target_key:
            self._last_target_key = target_key
        dt, dt_source = self._resolve_delta_time(raw)
        center_x, center_y = current_pos
        aim_x, aim_y = _aim_point(target, self.config.aim_ratio)
        error_x = float(aim_x - center_x)
        error_y = float(center_y - aim_y)
        target_extent = max(1.0, float(target.w), float(target.h))
        image_size = max(1.0, center_x * 2.0)
        output_x = self.axis_x.control_loop(error_x, dt, target_extent, image_size)
        output_y = self.axis_y.control_loop(error_y, dt, target_extent, image_size)
        move_kind = self.config.move_kind if self.config.move_kind in MOVE_KINDS else "raw"
        return MoveCommand(
            dx=output_x,
            dy=output_y,
            confidence=float(getattr(target, "score", 1.0)),
            reason="dynamic pid mouse strategy",
            move_kind=move_kind,
            move_ms=max(0, int(self.config.move_ms)),
            trace_ms=max(0, int(self.config.trace_ms)),
            debug={
                "stage": "dynamic_pid_pipeline",
                "algorithm": "dynamic_pid",
                "coordinate_y": "cartesian_up_positive",
                "aim_ratio": self.config.aim_ratio,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "raw_px_x": error_x,
                "raw_px_y": error_y,
                "dt_ms": dt * 1000.0,
                "dt_source": dt_source,
                "frame_id": raw.get("frame_id"),
                "capture_ts_ns": raw.get("capture_ts_ns"),
                "target_key": target_key,
                "target_changed": target_changed,
                "target_extent": target_extent,
                "image_size": image_size,
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
        self._last_target_key = ""

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
        error_change_tolerance=int(config.error_change_tolerance),
        smoothing_factor=config.smoothing_factor,
    )


def _aim_point(target: Target, aim_ratio: float) -> tuple[float, float]:
    ratio = max(0.0, min(100.0, float(aim_ratio))) / 100.0
    return float(target.cx), float(target.y) + float(target.h) * ratio
