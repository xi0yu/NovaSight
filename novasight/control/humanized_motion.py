from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, log2
from typing import Any


@dataclass(frozen=True, slots=True)
class HumanizedMotionInput:
    base_x: float
    base_y: float
    full_x: float
    full_y: float
    error_x_px: float
    error_y_px: float
    target_width_px: float
    target_id: int
    trigger_active: bool
    control_time_ms: float


@dataclass(frozen=True, slots=True)
class HumanizedMotionResult:
    x: float
    y: float
    telemetry: dict[str, Any]


class HumanizedMotionGenerator:
    """Turn a normalized trained or built-in trajectory into observation counts.

    The generator owns all profile phase state.  Callers provide the existing
    controller's full correction and demand; safety limiting remains outside
    this module.  With no profile or inactive trigger it is an exact identity.
    """

    def __init__(self, profile: dict[str, Any] | None) -> None:
        self.profile = profile if isinstance(profile, dict) else None
        self.reset()

    def reset(self) -> None:
        self._target_id: int | None = None
        self._segment_start_ms = 0.0
        self._last_progress = 0.0
        self._initial_full_x = 0.0
        self._initial_full_y = 0.0
        self._initial_magnitude = 0.0
        self._planned_duration_ms = 0.0
        self._last_side_position = 0.0

    def apply(self, value: HumanizedMotionInput) -> HumanizedMotionResult:
        if self.profile is None or not value.trigger_active:
            self.reset()
            return HumanizedMotionResult(value.base_x, value.base_y, {
                "humanized_motion_enabled": False,
                "humanized_motion_reason": "disabled_or_not_triggered",
            })
        if not all(isfinite(item) for item in (
            value.base_x, value.base_y, value.full_x, value.full_y,
            value.error_x_px, value.error_y_px, value.control_time_ms,
        )):
            self.reset()
            return HumanizedMotionResult(value.base_x, value.base_y, {
                "humanized_motion_enabled": False,
                "humanized_motion_reason": "non_finite_input",
            })

        current_magnitude = hypot(value.full_x, value.full_y)
        runtime = self.profile.get("runtime_parameters", {}) if self.profile else {}
        runtime = runtime if isinstance(runtime, dict) else {}
        micro_bypass = _bounded(runtime.get("micro_bypass_px", 0.0), 0.0, 1000.0)
        if hypot(value.error_x_px, value.error_y_px) <= micro_bypass:
            # A future departure from the micro zone must start a fresh path;
            # retaining an old segment here would resume it midway through.
            self.reset()
            return HumanizedMotionResult(value.base_x, value.base_y, {
                "humanized_motion_enabled": True,
                "humanized_motion_reason": "micro_bypass",
                "humanized_motion_phase": "closed_loop_correction",
                "humanized_motion_speed_curve_source": "micro_bypass",
                "humanized_motion_spatial_curve_source": "micro_bypass",
                "humanized_motion_progress": 1.0,
                "humanized_motion_curve_position": 1.0,
                "humanized_motion_curve_delta": 0.0,
                "humanized_motion_side_offset": 0.0,
                "humanized_motion_reference_x": value.base_x,
                "humanized_motion_reference_y": value.base_y,
                "humanized_motion_rebased": True,
            })
        dynamic_rebase = _bounded(runtime.get("dynamic_rebase_ratio", 0.25), 0.05, 1.0)
        rebase = (
            self._target_id != value.target_id
            or self._initial_magnitude <= 1e-6
            or value.control_time_ms < self._segment_start_ms
            or _direction_reversed(
                self._initial_full_x, self._initial_full_y, value.full_x, value.full_y
            )
            or current_magnitude > self._initial_magnitude * (1.0 + dynamic_rebase)
            or _endpoint_shift_ratio(
                self._initial_full_x,
                self._initial_full_y,
                value.full_x,
                value.full_y,
            ) > dynamic_rebase
        )
        if rebase:
            self._begin_segment(value)

        elapsed_ms = max(0.0, value.control_time_ms - self._segment_start_ms)
        progress = min(1.0, elapsed_ms / max(1.0, self._planned_duration_ms))
        curve = _profile_curve(self.profile, value.error_x_px, value.error_y_px, runtime)
        speed_curve_source = _profile_curve_source(
            self.profile, value.error_x_px, value.error_y_px, runtime
        )
        position = _interpolate_curve(curve, progress)
        previous_position = _interpolate_curve(curve, self._last_progress)
        delta_position = max(0.0, position - previous_position)
        side_curve = _profile_side_curve(self.profile, value.error_x_px, value.error_y_px)
        spatial_curve_source = _profile_side_curve_source(
            self.profile, value.error_x_px, value.error_y_px
        )
        side_position = _interpolate_curve(side_curve, progress)
        if not bool(runtime.get("spatial_curve_enabled", True)):
            side_position = 0.0
        side_position *= _bounded(runtime.get("side_scale", 1.0), 0.0, 4.0)
        side_position = max(-_bounded(runtime.get("max_side_ratio", 0.25), 0.0, 1.0),
                            min(_bounded(runtime.get("max_side_ratio", 0.25), 0.0, 1.0), side_position))
        fade_start = _bounded(runtime.get("near_fade_start_px", 0.0), 0.0, 5000.0)
        if fade_start > 0.0 and hypot(value.error_x_px, value.error_y_px) < fade_start:
            side_position *= hypot(value.error_x_px, value.error_y_px) / fade_start
        delta_side = side_position - self._last_side_position
        self._last_side_position = side_position
        self._last_progress = max(self._last_progress, progress)

        params = runtime
        correction_start = _bounded(params.get("correction_start_ratio", 0.82), 0.5, 1.0)
        correction_gain = _bounded(params.get("correction_gain", 0.35), 0.0, 1.0)
        terminal_gain = _bounded(params.get("terminal_feedback_gain", correction_gain), 0.0, 2.0)
        correction_weight = (
            0.0 if progress <= correction_start
            else min(1.0, (progress - correction_start) / max(1e-6, 1.0 - correction_start))
        )

        initial_unit_x = self._initial_full_x / max(1e-6, self._initial_magnitude)
        initial_unit_y = self._initial_full_y / max(1e-6, self._initial_magnitude)
        remaining_along = max(0.0, value.full_x * initial_unit_x + value.full_y * initial_unit_y)
        observed_progress_counts = self._initial_magnitude - remaining_along
        desired_progress_counts = self._initial_magnitude * position
        trajectory_error_counts = max(0.0, desired_progress_counts - observed_progress_counts)

        if progress >= 1.0:
            output_x = value.base_x * terminal_gain
            output_y = value.base_y * terminal_gain
            phase = "closed_loop_correction"
        else:
            direction_mag = max(1e-6, current_magnitude)
            # Track the trained cumulative curve against visually observed
            # progress.  This remains correct when latest-replace cancels an
            # older pending command because unsent motion stays in the error.
            step_cap = self._initial_magnitude * max(0.02, delta_position * 1.5)
            step_magnitude = min(trajectory_error_counts, step_cap)
            ux, uy = value.full_x / direction_mag, value.full_y / direction_mag
            # A learned side offset forms a 2-D Bezier-space path around the
            # direct line.  It is emitted as a delta per observation, never as
            # a device-side curve command.
            side_step = self._initial_magnitude * delta_side
            reference_x = ux * step_magnitude - uy * side_step
            reference_y = uy * step_magnitude + ux * side_step
            output_x = reference_x + value.base_x * correction_gain * correction_weight
            output_y = reference_y + value.base_y * correction_gain * correction_weight
            phase = _phase(progress)

        return HumanizedMotionResult(output_x, output_y, {
            "humanized_motion_enabled": True,
            "humanized_motion_phase": phase,
            "humanized_motion_speed_curve_source": speed_curve_source,
            "humanized_motion_spatial_curve_source": spatial_curve_source,
            "humanized_motion_progress": progress,
            "humanized_motion_curve_position": position,
            "humanized_motion_curve_delta": delta_position,
            "humanized_motion_side_offset": side_position,
            "humanized_motion_side_delta": delta_side,
            "humanized_motion_planned_duration_ms": self._planned_duration_ms,
            "humanized_motion_initial_counts": self._initial_magnitude,
            "humanized_motion_desired_progress_counts": desired_progress_counts,
            "humanized_motion_observed_progress_counts": observed_progress_counts,
            "humanized_motion_trajectory_error_counts": trajectory_error_counts,
            "humanized_motion_step_cap_counts": self._initial_magnitude * max(0.02, delta_position * 1.5),
            "humanized_motion_reference_x": output_x,
            "humanized_motion_reference_y": output_y,
            "humanized_motion_rebased": rebase,
        })

    def _begin_segment(self, value: HumanizedMotionInput) -> None:
        self._target_id = value.target_id
        self._segment_start_ms = max(0.0, value.control_time_ms)
        self._last_progress = 0.0
        self._initial_full_x = value.full_x
        self._initial_full_y = value.full_y
        self._initial_magnitude = hypot(value.full_x, value.full_y)
        distance_px = hypot(value.error_x_px, value.error_y_px)
        width_px = max(1.0, value.target_width_px)
        timing = self.profile.get("timing", {}) if self.profile else {}
        timing = timing if isinstance(timing, dict) else {}
        a_ms = _bounded(timing.get("fitts_a_ms", 90.0), 0.0, 1000.0)
        b_ms = _bounded(timing.get("fitts_b_ms", 85.0), 1.0, 1000.0)
        duration = a_ms + b_ms * log2(distance_px / width_px + 1.0)
        duration *= _direction_scale(self.profile, value.error_x_px, value.error_y_px)
        self._planned_duration_ms = min(1200.0, max(35.0, duration))
        self._last_side_position = 0.0


def _profile_curve(
    profile: dict[str, Any],
    error_x: float,
    error_y: float,
    runtime: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    raw = _profile_curve_raw(profile, error_x, error_y)
    values = (
        [min(1.0, max(0.0, float(item))) for item in raw if isinstance(item, int | float)]
        if isinstance(raw, list)
        else []
    )
    if len(values) >= 2:
        values[0] = 0.0
        for index in range(1, len(values)):
            values[index] = max(values[index - 1], values[index])
        values[-1] = 1.0
        return tuple(values)
    if runtime and runtime.get("minimum_jerk_fallback", True):
        return tuple(
            10 * t**3 - 15 * t**4 + 6 * t**5
            for t in (index / 8.0 for index in range(9))
        )
    return (0.0, 1.0)


def _profile_curve_source(
    profile: dict[str, Any],
    error_x: float,
    error_y: float,
    runtime: dict[str, Any] | None = None,
) -> str:
    raw = _profile_curve_raw(profile, error_x, error_y)
    valid_points = (
        [item for item in raw if isinstance(item, int | float)]
        if isinstance(raw, list)
        else []
    )
    if len(valid_points) >= 2:
        return "trained_progress"
    if runtime and runtime.get("minimum_jerk_fallback", True):
        return "minimum_jerk"
    return "linear"


def _profile_curve_raw(profile: dict[str, Any], error_x: float, error_y: float) -> Any:
    selected = _distance_profile(profile, error_x, error_y)
    return selected.get("progress_curve") or profile.get("progress_curve")


def _interpolate_curve(curve: tuple[float, ...], progress: float) -> float:
    position = min(1.0, max(0.0, progress)) * (len(curve) - 1)
    left = min(len(curve) - 1, int(position))
    right = min(len(curve) - 1, left + 1)
    weight = position - left
    return curve[left] + (curve[right] - curve[left]) * weight


def _profile_side_curve(
    profile: dict[str, Any], error_x: float = 0.0, error_y: float = 0.0
) -> tuple[float, ...]:
    """Return normalized lateral offset; supports a cubic Bezier profile."""
    selected = _distance_profile(profile, error_x, error_y)
    raw = selected.get("side_offset_curve") if isinstance(selected, dict) else None
    raw = raw or profile.get("side_offset_curve")
    if isinstance(raw, dict):
        points = raw.get("control_points")
        if isinstance(points, list) and len(points) == 4:
            values = [_bounded(item, -1.0, 1.0) for item in points]
            values[0] = 0.0
            values[-1] = 0.0
            return tuple(_bezier(values, i / 8.0) for i in range(9))
    if isinstance(raw, list) and len(raw) >= 2:
        return tuple(_bounded(item, -1.0, 1.0) for item in raw)
    return (0.0, 0.0)


def _profile_side_curve_source(
    profile: dict[str, Any], error_x: float = 0.0, error_y: float = 0.0
) -> str:
    selected = _distance_profile(profile, error_x, error_y)
    raw = selected.get("side_offset_curve") if isinstance(selected, dict) else None
    raw = raw or profile.get("side_offset_curve")
    if isinstance(raw, dict):
        points = raw.get("control_points")
        if isinstance(points, list) and len(points) == 4:
            return "cubic_bezier"
    if isinstance(raw, list) and len(raw) >= 2:
        return "sampled_side_curve"
    return "none"


def _distance_profile(
    profile: dict[str, Any], error_x: float, error_y: float
) -> dict[str, Any]:
    profiles = profile.get("distance_profiles", {})
    if not isinstance(profiles, dict):
        return {}
    distance = hypot(error_x, error_y)
    key = (
        "micro"
        if distance < 50.0
        else "near"
        if distance < 150.0
        else "mid"
        if distance < 350.0
        else "far"
    )
    selected = profiles.get(key, {})
    return selected if isinstance(selected, dict) else {}


def _bezier(points: list[float], t: float) -> float:
    u = 1.0 - t
    return u**3 * points[0] + 3.0 * u**2 * t * points[1] + 3.0 * u * t**2 * points[2] + t**3 * points[3]


def _direction_scale(profile: dict[str, Any] | None, x: float, y: float) -> float:
    if not profile:
        return 1.0
    scales = profile.get("direction_duration_scales", {})
    if not isinstance(scales, dict):
        return 1.0
    if abs(x) >= abs(y) * 1.5:
        key = "right" if x >= 0.0 else "left"
    elif abs(y) >= abs(x) * 1.5:
        key = "down" if y >= 0.0 else "up"
    else:
        vertical = "down" if y >= 0.0 else "up"
        horizontal = "right" if x >= 0.0 else "left"
        key = f"{vertical}_{horizontal}"
    # Profiles before v3 stored every diagonal in one shared bucket.
    fallback = scales.get("diagonal", 1.0) if "_" in key else 1.0
    return _bounded(scales.get(key, fallback), 0.65, 1.45)


def _direction_reversed(initial_x: float, initial_y: float, x: float, y: float) -> bool:
    initial_mag = hypot(initial_x, initial_y)
    current_mag = hypot(x, y)
    if min(initial_mag, current_mag) <= 1e-6:
        return False
    return (initial_x * x + initial_y * y) / (initial_mag * current_mag) < 0.25


def _endpoint_shift_ratio(initial_x: float, initial_y: float, x: float, y: float) -> float:
    """Measure endpoint drift perpendicular to the active segment."""

    initial_mag = hypot(initial_x, initial_y)
    if initial_mag <= 1e-6:
        return 0.0
    unit_x = initial_x / initial_mag
    unit_y = initial_y / initial_mag
    lateral = x * (-unit_y) + y * unit_x
    return abs(lateral) / initial_mag


def _phase(progress: float) -> str:
    if progress < 0.2:
        return "startup"
    if progress < 0.55:
        return "acceleration"
    if progress < 0.78:
        return "braking"
    return "fine_correction"


def _bounded(value: object, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lower
    if not isfinite(number):
        number = lower
    return min(upper, max(lower, number))
