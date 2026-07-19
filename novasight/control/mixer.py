from __future__ import annotations

from dataclasses import dataclass

from .recoil import RecoilDecision


@dataclass(frozen=True, slots=True)
class ControlMix:
    tracking_x: float
    tracking_y: float
    recoil_y: float
    final_x: float
    final_y: float
    recoil_active: bool


class ControlMixer:
    """Single arbitration point for tracking and independent recoil output."""

    def mix(self, tracking_x: float, tracking_y: float, recoil: RecoilDecision) -> ControlMix:
        # Recoil owns the downward component while firing so target tracking
        # cannot double the compensation.  Upward tracking remains available
        # to correct an overshoot as the recoil gate closes.
        state = recoil.state.value
        engaged = state in {"STARTUP", "ACTIVE", "HOLD", "BRAKE"} and (
            recoil.gate > 0.0
            or recoil.final_rate_counts_s > 0.0
            or recoil.residual_counts_y > 0.0
        )
        if engaged:
            # Positive values are physical downward motion.  Do not stack two
            # downward demands; upward tracking remains able to cancel recoil.
            tracking_down = max(0.0, float(tracking_y))
            final_y = max(float(recoil.emitted_counts_y), tracking_down) + min(
                0.0, float(tracking_y)
            )
        else:
            # Recoil staleness disables only the independent branch.  The
            # tracking branch keeps its own freshness and target-loss policy.
            final_y = float(tracking_y)
        return ControlMix(float(tracking_x), float(tracking_y), float(recoil.emitted_counts_y),
                          float(tracking_x), final_y, engaged)


__all__ = ["ControlMix", "ControlMixer"]
