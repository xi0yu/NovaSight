from __future__ import annotations

from dataclasses import replace
import threading
import time

import pytest

from novasight.config import RuntimeConfig
from novasight.control import ControlOutput, ScheduleDecision
from novasight.executors import ExecutionResult, ExecutorRegistry, MouseCommandExecutor


class _RecordingExecutor:
    executor_id = "kmnet"

    def __init__(self) -> None:
        self.outputs: list[ControlOutput] = []
        self.trigger_active = True

    def available(self) -> bool:
        return True

    def execute(self, output: ControlOutput) -> ExecutionResult:
        self.outputs.append(output)
        return ExecutionResult(self.executor_id, True, output, "sent")

    def read_buttons(self) -> dict[str, object]:
        return {
            "available": True,
            "left": self.trigger_active,
            "right": False,
        }


class _SlowTriggerExecutor(_RecordingExecutor):
    def read_buttons(self) -> dict[str, object]:
        time.sleep(0.020)
        return super().read_buttons()


def _command(
    generation: int,
    *,
    trigger_active: bool = True,
    expires_ts_ns: int | None = None,
) -> ControlOutput:
    return ControlOutput(
        dx=5,
        dy=1,
        action="move",
        confidence=1.0,
        source_id="test",
        accepted=True,
        clipped=False,
        reason="test",
        source_frame_id=generation,
        trajectory_generation=generation,
        trigger_required=True,
        trigger_active=trigger_active,
        command_expires_ts_ns=(
            time.monotonic_ns() + 1_000_000_000 if expires_ts_ns is None else expires_ts_ns
        ),
    )


def test_mouse_command_executor_rechecks_trigger_freshness_and_generation() -> None:
    device = _RecordingExecutor()
    executor = MouseCommandExecutor()

    trigger_blocked = executor.execute(
        executor=device,
        command=_command(1, trigger_active=False),
    )
    stale_blocked = executor.execute(
        executor=device,
        command=_command(1, expires_ts_ns=1),
    )
    sent = executor.execute(executor=device, command=_command(1))
    replay_blocked = executor.execute(executor=device, command=_command(1))

    assert trigger_blocked.metadata["block_reason"] == "TRIGGER_INACTIVE"
    assert stale_blocked.metadata["block_reason"] == "COMMAND_STALE"
    assert sent.sent is True
    assert replay_blocked.metadata["block_reason"] == "COMMAND_ACTUATION_NOT_NEWER"
    assert [(output.dx, output.dy) for output in device.outputs] == [(5, 1)]

    executor.reset()
    restarted = executor.execute(executor=device, command=_command(1))

    assert restarted.sent is True
    assert [(output.dx, output.dy) for output in device.outputs] == [(5, 1), (5, 1)]


def test_mouse_command_executor_rechecks_live_hardware_trigger_before_send() -> None:
    device = _RecordingExecutor()
    device.trigger_active = False
    executor = MouseCommandExecutor()

    blocked = executor.execute(
        executor=device,
        command=_command(1, trigger_active=True),
    )

    assert blocked.sent is False
    assert blocked.metadata["block_reason"] == "TRIGGER_INACTIVE"
    assert device.outputs == []

    device.read_buttons = lambda: {  # type: ignore[method-assign]
        "available": True,
        "left": False,
        "right": True,
    }
    tracking = executor.execute(executor=device, command=_command(1))
    executor.reset()
    recoil = executor.execute(
        executor=device,
        command=replace(_command(1), left_trigger_required=True),
    )

    assert tracking.sent is True
    assert recoil.sent is False
    assert recoil.metadata["block_reason"] == "TRIGGER_INACTIVE"


def test_mouse_command_executor_rechecks_freshness_after_live_trigger_read() -> None:
    device = _SlowTriggerExecutor()
    executor = MouseCommandExecutor()

    blocked = executor.execute(
        executor=device,
        command=_command(
            1,
            expires_ts_ns=time.monotonic_ns() + 5_000_000,
        ),
    )

    assert blocked.sent is False
    assert blocked.metadata["block_reason"] == "COMMAND_STALE"
    assert device.outputs == []


def test_mouse_command_executor_requires_explicit_trigger_contract() -> None:
    device = _RecordingExecutor()
    executor = MouseCommandExecutor()
    command = replace(_command(1), trigger_required=None)

    blocked = executor.execute(executor=device, command=command)

    assert blocked.sent is False
    assert blocked.metadata["block_reason"] == "TRIGGER_REQUIREMENT_MISSING"
    assert device.outputs == []


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        (replace(_command(1), dx=1.5), "COMMAND_COUNTS_NOT_INTEGER"),
        (replace(_command(1), dx=32_768), "COMMAND_COUNTS_OUT_OF_DEVICE_RANGE"),
    ],
)
def test_mouse_command_executor_rejects_invalid_device_counts(
    command: ControlOutput,
    reason: str,
) -> None:
    device = _RecordingExecutor()
    executor = MouseCommandExecutor()

    blocked = executor.execute(executor=device, command=command)

    assert blocked.sent is False
    assert blocked.metadata["block_reason"] == reason
    assert device.outputs == []


class _BlockingTickScheduler:
    def __init__(self, output: ControlOutput, ready: threading.Event) -> None:
        self.output = output
        self.ready = ready

    def tick(self, *, now_s: float | None = None) -> ScheduleDecision:
        del now_s
        self.ready.set()
        return ScheduleDecision(self.output, {"action": "emit_step"})

    def record_execution_result(self, **_: object) -> dict[str, object]:
        return {"recorded": True}

    def status(self) -> dict[str, object]:
        return {"enabled": True}

    def clear(self, reason: str) -> None:
        del reason


def test_single_command_mode_unconditionally_discards_injected_scheduler() -> None:
    device = _RecordingExecutor()
    scheduler = _BlockingTickScheduler(_command(1), threading.Event())

    registry = ExecutorRegistry(
        [device],
        scheduler=scheduler,  # type: ignore[arg-type]
        direct_output=False,
        single_command_per_observation=True,
    )

    assert registry.scheduler is None
    assert registry.direct_output is True
    assert registry.single_command_per_observation is True


def test_single_command_is_discarded_when_algorithm_switches_before_send() -> None:
    device = _RecordingExecutor()
    registry = ExecutorRegistry(
        [device],
        direct_output=True,
        single_command_per_observation=True,
    )
    direct_ready = threading.Event()
    original_execute_direct = registry._execute_direct

    def execute_direct_after_snapshot(*args: object, **kwargs: object) -> ExecutionResult:
        direct_ready.set()
        return original_execute_direct(*args, **kwargs)  # type: ignore[arg-type]

    registry._execute_direct = execute_direct_after_snapshot  # type: ignore[method-assign]
    registry._executor_lock.acquire()
    result_holder: list[ExecutionResult] = []
    thread = threading.Thread(target=lambda: result_holder.append(registry.execute(_command(10))))

    try:
        thread.start()
        assert direct_ready.wait(timeout=1.0)
        updated = RuntimeConfig()
        registry.update_runtime_config(updated)
    finally:
        registry._executor_lock.release()
        thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert len(result_holder) == 1
    assert result_holder[0].sent is False
    assert result_holder[0].metadata["action"] == "executor_superseded"
    assert device.outputs == []


def test_scheduler_tick_is_discarded_when_algorithm_switches_before_device_send() -> None:
    device = _RecordingExecutor()
    tick_ready = threading.Event()
    old_output = _command(10)
    scheduler = _BlockingTickScheduler(old_output, tick_ready)
    registry = ExecutorRegistry(
        [device],
        scheduler=scheduler,  # type: ignore[arg-type]
    )
    registry._executor_lock.acquire()
    result_holder: list[ExecutionResult] = []
    thread = threading.Thread(target=lambda: result_holder.append(registry.tick_pending()))

    try:
        thread.start()
        assert tick_ready.wait(timeout=1.0)
        updated = RuntimeConfig()
        updated.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
        registry.update_runtime_config(updated)
    finally:
        registry._executor_lock.release()
        thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert len(result_holder) == 1
    assert result_holder[0].sent is False
    assert result_holder[0].metadata["action"] == "scheduler_superseded"
    assert device.outputs == []
