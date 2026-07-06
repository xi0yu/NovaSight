"""Tests for hardware heartbeat, control strategies, and the
throttling coalescer used to bridge the inference / control loops.

Each test focuses on one observable contract:
- Heartbeat suspends after a stale window.
- PIDStrategy clamps the per-axis move and returns target score as
  confidence.
- PredictiveStrategy drops output when the hardware trigger is
  inactive and uses the last-observed center as the velocity source.
- ControlCommandCoalescer emits at most one intent per min_interval
  window and tags throttled emissions with the dropped count.
"""
from __future__ import annotations

from novasight.contracts import ControlIntent, Detection
from novasight.control import ControlCommandCoalescer, PIDStrategy, PredictiveStrategy
from novasight.hardware import BoxInputState, HardwareHeartbeat


def _target(
    x: float = 100, y: float = 100, w: float = 0, h: float = 0, score: float = 0.9
) -> Detection:
    return Detection(cls=0, score=score, x=x, y=y, w=w, h=h)


def _intent(dx: float, dy: float, reason: str = "test") -> ControlIntent:
    return ControlIntent(
        dx=dx, dy=dy, action="move", confidence=1.0,
        reason=reason, source_id="control.test",
    )


def test_hardware_heartbeat_suspends_after_timeout() -> None:
    heartbeat = HardwareHeartbeat(timeout_s=0.05)

    assert heartbeat.should_suspend(now_s=1.0) is True
    heartbeat.mark_seen(now_s=1.0)

    assert heartbeat.should_suspend(now_s=1.02) is False
    assert heartbeat.should_suspend(now_s=1.10) is True


def test_pid_strategy_clamps_to_move_limit_per_axis() -> None:
    strategy = PIDStrategy(
        kp_x=2.0,
        kp_y=0.5,
        ki=0.0,
        kd=0.0,
        move_limit=10,
    )

    command = strategy.calculate(_target(120, 110), (100, 100), BoxInputState(left=True))

    assert command.dx == 10
    assert command.dy == -10  # cartesian-up: target above center => dy < 0
    assert command.confidence == 0.9


def test_predictive_strategy_drops_output_without_hardware_trigger() -> None:
    strategy = PredictiveStrategy(lead_factor=1.0)

    command = strategy.calculate(_target(100, 100), (100, 100), BoxInputState())

    assert command.dx == 0
    assert command.dy == 0
    assert command.confidence == 0


def test_predictive_strategy_leads_moving_target() -> None:
    strategy = PredictiveStrategy(lead_factor=1.0)
    trigger = BoxInputState(left=True)

    # First call seeds the strategy with the target's current center so no
    # velocity exists yet, and the target already sits on the center axis.
    first = strategy.calculate(_target(100, 100), (100, 100), trigger)
    # Second call observes a target that moved right by 20px; the lead
    # projects the move ahead and emits positive dx.
    second = strategy.calculate(_target(120, 100), (100, 100), trigger)

    assert first.dx == 0
    assert first.dy == 0
    assert second.dx > 0


def test_control_command_coalescer_emits_one_per_interval() -> None:
    coalescer = ControlCommandCoalescer(min_interval_s=0.01)
    first = _intent(1, 2, "first")
    second = _intent(3, 4, "second")

    first_emitted = coalescer.push(first, now_s=1.0)
    second_dropped = coalescer.push(second, now_s=1.005)
    third_merged = coalescer.push(second, now_s=1.02)

    assert first_emitted is not None
    assert first_emitted.dx == 1
    assert second_dropped is None  # within the 10ms window: throttled
    assert third_merged is not None
    assert third_merged.dx == 3
    assert "dropped=1" in third_merged.reason
