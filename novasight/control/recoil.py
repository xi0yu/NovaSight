from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, trunc


@dataclass(frozen=True, slots=True)
class FixedRecoilConfig:
    enabled: bool = False
    start_delay_ms: float = 0.0
    y_counts_per_observation: float = 0.0
    invert_y: bool = False


@dataclass(frozen=True, slots=True)
class FixedRecoilDecision:
    active: bool
    requested_counts_y: float
    emitted_counts_y: int
    residual_counts_y: float
    block_reason: str


class FixedRecoilController:
    """Emit one fixed reverse-Y contribution for each accepted observation."""

    def __init__(self, config: FixedRecoilConfig) -> None:
        _validate_config(config)
        self.config = config
        self._residual_counts_y = 0.0

    def reset(self) -> None:
        self._residual_counts_y = 0.0

    def calculate(
        self,
        *,
        left_trigger_active: bool,
        left_trigger_hold_ms: float,
    ) -> FixedRecoilDecision:
        block_reason = self._block_reason(
            left_trigger_active=left_trigger_active,
            left_trigger_hold_ms=left_trigger_hold_ms,
        )
        if block_reason:
            self.reset()
            return FixedRecoilDecision(
                active=False,
                requested_counts_y=0.0,
                emitted_counts_y=0,
                residual_counts_y=0.0,
                block_reason=block_reason,
            )

        direction = -1.0 if self.config.invert_y else 1.0
        requested = direction * self.config.y_counts_per_observation
        accumulator = self._residual_counts_y + requested
        emitted = trunc(accumulator)
        self._residual_counts_y = accumulator - emitted
        return FixedRecoilDecision(
            active=True,
            requested_counts_y=requested,
            emitted_counts_y=emitted,
            residual_counts_y=self._residual_counts_y,
            block_reason="",
        )

    def _block_reason(
        self,
        *,
        left_trigger_active: bool,
        left_trigger_hold_ms: float,
    ) -> str:
        if not self.config.enabled:
            return "RECOIL_DISABLED"
        if not left_trigger_active:
            return "LEFT_TRIGGER_INACTIVE"
        if not isfinite(float(left_trigger_hold_ms)) or float(left_trigger_hold_ms) < 0.0:
            return "LEFT_TRIGGER_HOLD_INVALID"
        if float(left_trigger_hold_ms) < self.config.start_delay_ms:
            return "RECOIL_START_DELAY"
        if self.config.y_counts_per_observation <= 0.0:
            return "RECOIL_COUNTS_ZERO"
        return ""


def _validate_config(config: FixedRecoilConfig) -> None:
    for name, value in (
        ("start_delay_ms", config.start_delay_ms),
        ("y_counts_per_observation", config.y_counts_per_observation),
    ):
        if not isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"fixed recoil {name} must be finite and >= 0")


__all__ = [
    "FixedRecoilConfig",
    "FixedRecoilController",
    "FixedRecoilDecision",
]
