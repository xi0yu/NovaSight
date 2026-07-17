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
    left_trigger_active: bool
    trigger_hold_ms: float


@dataclass(frozen=True, slots=True)
class HumanizedMotionResult:
    x: float
    y: float
    telemetry: dict[str, Any]


class HumanizedMotionGenerator:
    """Turns a trained normalized movement curve into per-observation counts.

    The generator owns all profile phase state.  Callers provide the existing
    controller's full correction and demand; safety limiting remains outside
    this module.  With no profile or inactive trigger it is an exact identity.
    """

    def __init__(self, profile: dict[str, Any] | None) -> None:
        self.profile = profile if isinstance(profile, dict) else None
        self.reset()

    def reset(self) -> None:
        self._target_id: int | None = None
        self._segment_hold_start_ms = 0.0
        self._last_progress = 0.0
        self._initial_full_x = 0.0
        self._initial_full_y = 0.0
        self._initial_magnitude = 0.0
        self._planned_duration_ms = 0.0

    def apply(self, value: HumanizedMotionInput) -> HumanizedMotionResult:
        if self.profile is None or not value.trigger_active or not value.left_trigger_active:
            self.reset()
            return HumanizedMotionResult(value.base_x, value.base_y, {
                "humanized_motion_enabled": False,
                "humanized_motion_reason": "disabled_or_not_triggered",
            })
        if not all(isfinite(item) for item in (
            value.base_x, value.base_y, value.full_x, value.full_y,
            value.error_x_px, value.error_y_px, value.trigger_hold_ms,
        )):
            self.reset()
            return HumanizedMotionResult(value.base_x, value.base_y, {
                "humanized_motion_enabled": False,
                "humanized_motion_reason": "non_finite_input",
            })

        current_magnitude = hypot(value.full_x, value.full_y)
        rebase = (
            self._target_id != value.target_id
            or self._initial_magnitude <= 1e-6
            or value.trigger_hold_ms < self._segment_hold_start_ms
            or _direction_reversed(
                self._initial_full_x, self._initial_full_y, value.full_x, value.full_y
            )
            or current_magnitude > self._initial_magnitude * 1.35
        )
        if rebase:
            self._begin_segment(value)

        elapsed_ms = max(0.0, value.trigger_hold_ms - self._segment_hold_start_ms)
        progress = min(1.0, elapsed_ms / max(1.0, self._planned_duration_ms))
        curve = _profile_curve(self.profile, value.error_x_px, value.error_y_px)
        position = _interpolate_curve(curve, progress)
        previous_position = _interpolate_curve(curve, self._last_progress)
        delta_position = max(0.0, position - previous_position)
        self._last_progress = max(self._last_progress, progress)

        params = self.profile.get("runtime_parameters", {})
        params = params if isinstance(params, dict) else {}
        correction_start = _bounded(params.get("correction_start_ratio", 0.82), 0.5, 1.0)
        correction_gain = _bounded(params.get("correction_gain", 0.35), 0.0, 1.0)
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
            output_x = value.base_x * max(correction_gain, correction_weight)
            output_y = value.base_y * max(correction_gain, correction_weight)
            phase = "closed_loop_correction"
        else:
            direction_mag = max(1e-6, current_magnitude)
            # Track the trained cumulative curve against visually observed
            # progress.  This remains correct when latest-replace cancels an
            # older pending command because unsent motion stays in the error.
            step_cap = self._initial_magnitude * max(0.02, delta_position * 1.5)
            step_magnitude = min(trajectory_error_counts, step_cap)
            reference_x = value.full_x / direction_mag * step_magnitude
            reference_y = value.full_y / direction_mag * step_magnitude
            output_x = reference_x + value.base_x * correction_gain * correction_weight
            output_y = reference_y + value.base_y * correction_gain * correction_weight
            phase = _phase(progress)

        return HumanizedMotionResult(output_x, output_y, {
            "humanized_motion_enabled": True,
            "humanized_motion_phase": phase,
            "humanized_motion_progress": progress,
            "humanized_motion_curve_position": position,
            "humanized_motion_curve_delta": delta_position,
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
        self._segment_hold_start_ms = max(0.0, value.trigger_hold_ms)
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


def _profile_curve(profile: dict[str, Any], error_x: float, error_y: float) -> tuple[float, ...]:
    curves = profile.get("distance_profiles", {})
    curves = curves if isinstance(curves, dict) else {}
    distance = hypot(error_x, error_y)
    key = "micro" if distance < 50.0 else "near" if distance < 150.0 else "mid" if distance < 350.0 else "far"
    selected = curves.get(key, {})
    selected = selected if isinstance(selected, dict) else {}
    raw = selected.get("progress_curve") or profile.get("progress_curve")
    if not isinstance(raw, list) or len(raw) < 2:
        return (0.0, 0.04, 0.13, 0.27, 0.45, 0.64, 0.79, 0.90, 0.96, 1.0)
    values = [min(1.0, max(0.0, float(item))) for item in raw if isinstance(item, int | float)]
    if len(values) < 2:
        return (0.0, 1.0)
    values[0] = 0.0
    for index in range(1, len(values)):
        values[index] = max(values[index - 1], values[index])
    values[-1] = 1.0
    return tuple(values)


def _interpolate_curve(curve: tuple[float, ...], progress: float) -> float:
    position = min(1.0, max(0.0, progress)) * (len(curve) - 1)
    left = min(len(curve) - 1, int(position))
    right = min(len(curve) - 1, left + 1)
    weight = position - left
    return curve[left] + (curve[right] - curve[left]) * weight


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
