from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, trunc


class RecoilState(str, Enum):
    IDLE = "IDLE"
    STARTUP = "STARTUP"
    ACTIVE = "ACTIVE"
    HOLD = "HOLD"
    BRAKE = "BRAKE"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class TargetRelativeRecoilConfig:
    enabled: bool = False
    base_rate_counts_s: float = 0.0
    max_rate_counts_s: float = 0.0
    startup_ms: float = 35.0
    positive_deadzone_norm: float = 0.04
    negative_deadzone_norm: float = 0.04
    full_brake_error_norm: float = 0.12
    fast_add_gain_counts_s: float = 0.0
    max_fast_add_ratio: float = 0.30
    stale_threshold_ms: float = 55.0


@dataclass(frozen=True, slots=True)
class RecoilInput:
    firing: bool
    now_ns: int
    dt_s: float
    target_valid: bool
    target_id: object | None = None
    source_generation: object | None = None
    observation_age_ms: float = 0.0
    error_y_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class RecoilDecision:
    state: RecoilState
    base_rate_counts_s: float
    fast_add_rate_counts_s: float
    gate: float
    final_rate_counts_s: float
    requested_counts_y: float
    emitted_counts_y: int
    residual_counts_y: float
    block_reason: str


class TargetRelativeRecoilController:
    """Time-integrated, target-aware positive-Y recoil feed-forward."""

    def __init__(self, config: TargetRelativeRecoilConfig) -> None:
        _validate_config(config)
        self.config = config
        self.state = RecoilState.IDLE
        self._fire_start_ns: int | None = None
        self._residual = 0.0
        self._target_key: object | None = None

    def reset(self) -> None:
        self.state = RecoilState.IDLE
        self._fire_start_ns = None
        self._residual = 0.0
        self._target_key = None

    def calculate(self, value: RecoilInput) -> RecoilDecision:
        c = self.config
        if not c.enabled:
            self.reset()
            return self._decision("RECOIL_DISABLED")
        if not value.firing:
            self.reset()
            return self._decision("FIRING_INACTIVE")
        if not isfinite(float(value.dt_s)) or value.dt_s < 0.0:
            self.reset()
            return self._decision("DT_INVALID")
        if not isfinite(float(value.observation_age_ms)) or value.observation_age_ms < 0.0:
            self.reset()
            return self._decision("OBSERVATION_AGE_INVALID")
        if value.observation_age_ms > c.stale_threshold_ms or not value.target_valid:
            self.state = RecoilState.STALE
            self._residual = 0.0
            return self._decision("TARGET_STALE" if value.target_valid else "TARGET_INVALID")
        if not isfinite(float(value.error_y_norm)):
            self.state = RecoilState.STALE
            self._residual = 0.0
            return self._decision("ERROR_INVALID")
        if self._fire_start_ns is None:
            self._fire_start_ns = int(value.now_ns)
        # Target switches do not reset the continuous fire timing.
        target_key = value.target_id
        if self._target_key is not None and target_key != self._target_key:
            # Clear fractional debt at a switch, but deliberately retain the
            # continuous fire-start clock so startup is not replayed.
            self._residual = 0.0
        self._target_key = target_key
        elapsed_ms = max(0.0, (int(value.now_ns) - self._fire_start_ns) / 1e6)
        startup_gate = (
            1.0
            if c.startup_ms <= 0.0
            else min(1.0, (elapsed_ms + float(value.dt_s) * 1000.0) / c.startup_ms)
        )
        error = float(value.error_y_norm)
        if error <= -c.full_brake_error_norm:
            gate = 0.0
        elif error < -c.negative_deadzone_norm:
            gate = 1.0 - (
                (-error - c.negative_deadzone_norm)
                / (c.full_brake_error_norm - c.negative_deadzone_norm)
            )
        else:
            gate = 1.0
        if error < -c.negative_deadzone_norm:
            self.state = RecoilState.BRAKE
        elif startup_gate < 1.0:
            self.state = RecoilState.STARTUP
        elif error <= c.positive_deadzone_norm:
            self.state = RecoilState.HOLD
        else:
            self.state = RecoilState.ACTIVE
        fast = 0.0
        if error > c.positive_deadzone_norm:
            fast = min(
                c.fast_add_gain_counts_s * (error - c.positive_deadzone_norm),
                c.max_rate_counts_s * c.max_fast_add_ratio,
            )
        rate = min(
            c.max_rate_counts_s,
            max(0.0, (c.base_rate_counts_s + fast) * gate * startup_gate),
        )
        requested = rate * float(value.dt_s)
        acc = self._residual + requested
        emitted = max(0, trunc(acc))
        self._residual = acc - emitted
        block_reason = (
            "POSITION_BRAKE"
            if gate <= 0.0
            else "RECOIL_RATE_ZERO"
            if rate <= 0.0
            else ""
        )
        return RecoilDecision(
            self.state,
            c.base_rate_counts_s,
            fast,
            gate,
            rate,
            requested,
            emitted,
            self._residual,
            block_reason,
        )

    def _decision(self, reason: str) -> RecoilDecision:
        return RecoilDecision(
            self.state,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            self._residual,
            reason,
        )


def _validate_config(c: TargetRelativeRecoilConfig) -> None:
    nonnegative = (
        "base_rate_counts_s",
        "max_rate_counts_s",
        "startup_ms",
        "positive_deadzone_norm",
        "negative_deadzone_norm",
        "full_brake_error_norm",
        "fast_add_gain_counts_s",
        "max_fast_add_ratio",
        "stale_threshold_ms",
    )
    for name in nonnegative:
        value = float(getattr(c, name))
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"recoil {name} must be finite and >= 0")
    if c.max_rate_counts_s < c.base_rate_counts_s:
        raise ValueError("recoil max_rate_counts_s must be >= base_rate_counts_s")
    for name in (
        "positive_deadzone_norm",
        "negative_deadzone_norm",
        "full_brake_error_norm",
        "max_fast_add_ratio",
    ):
        if float(getattr(c, name)) > 1.0:
            raise ValueError(f"recoil {name} must be <= 1")
    if c.full_brake_error_norm <= c.negative_deadzone_norm:
        raise ValueError("recoil full_brake_error_norm must be > negative_deadzone_norm")


__all__ = [
    "RecoilState",
    "TargetRelativeRecoilConfig",
    "RecoilInput",
    "RecoilDecision",
    "TargetRelativeRecoilController",
]
