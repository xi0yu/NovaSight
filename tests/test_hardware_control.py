from __future__ import annotations

from novasight.control import ControlCommandCoalescer, PIDStrategy, PredictiveStrategy
from novasight.hardware import BoxInputState, HardwareHeartbeat
from novasight.plugins import ControlIntent, Detection


def _target(x: float = 100, y: float = 90) -> Detection:
    return Detection(cls=0, score=0.9, x=x, y=y, w=20, h=20)


def test_hardware_heartbeat_suspends_after_timeout() -> None:
    heartbeat = HardwareHeartbeat(timeout_s=0.05)

    assert heartbeat.should_suspend(now_s=1.0) is True
    heartbeat.mark_seen(now_s=1.0)

    assert heartbeat.should_suspend(now_s=1.02) is False
    assert heartbeat.should_suspend(now_s=1.10) is True


def test_pid_strategy_requires_hardware_trigger_and_limits_integral() -> None:
    strategy = PIDStrategy(kp=1.0, ki=1.0, kd=0.0, integral_limit=5)

    inactive = strategy.calculate(_target(), (50, 50), BoxInputState())
    active = strategy.calculate(_target(), (50, 50), BoxInputState(left=True))

    assert inactive.confidence == 0
    assert active.dx > 0
    assert active.dy > 0
    assert active.confidence == 0.9


def test_predictive_strategy_leads_moving_target() -> None:
    strategy = PredictiveStrategy(lead_factor=1.0)
    trigger = BoxInputState(left=True)

    first = strategy.calculate(_target(100, 90), (110, 100), trigger)
    second = strategy.calculate(_target(120, 90), (110, 100), trigger)

    assert first.dx == 0
    assert second.dx > 20


def test_control_command_coalescer_merges_commands_within_interval() -> None:
    coalescer = ControlCommandCoalescer(min_interval_s=0.01)
    first = ControlIntent(1, 2, "move", 1.0, "first", "test")
    second = ControlIntent(3, 4, "move", 1.0, "second", "test")

    assert coalescer.push(first, now_s=1.0) is not None
    assert coalescer.push(first, now_s=1.005) is None
    merged = coalescer.push(second, now_s=1.02)

    assert merged is not None
    assert merged.dx == 4
    assert merged.dy == 6
