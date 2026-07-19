from __future__ import annotations

from _thread import LockType
from dataclasses import replace
import threading
import time

from novasight.control import MAX_ABS_MOUSE_MOVE_COUNT, ControlOutput
from novasight.executors.contracts import ExecutionResult, Executor


class MouseCommandExecutor:
    """Validate and send one current-observation integer command.

    It retains only the latest consumed generation for replay rejection. It has
    no queue, movement target, trajectory, or cross-frame count debt.
    """

    MAX_ABS_DEVICE_COUNT = MAX_ABS_MOUSE_MOVE_COUNT

    def __init__(self, lock: LockType | None = None) -> None:
        self._lock = lock or threading.Lock()
        self._last_generation: int | None = None

    def reset(self) -> None:
        with self._lock:
            self._last_generation = None

    def execute(
        self,
        *,
        executor: Executor,
        command: ControlOutput,
    ) -> ExecutionResult:
        with self._lock:
            return self.execute_locked(executor=executor, command=command)

    def execute_locked(
        self,
        *,
        executor: Executor,
        command: ControlOutput,
    ) -> ExecutionResult:
        """Execute while the caller already owns the shared device lock."""

        checked_ts_ns = time.monotonic_ns()
        block_reason = self._contract_block_reason(executor, command)
        if block_reason:
            return self._blocked_result(
                executor=executor,
                command=command,
                reason=block_reason,
                checked_ts_ns=checked_ts_ns,
            )

        # A live trigger read can block. Re-sample freshness after it, directly
        # before the single device call, so an expired command never leaks out.
        send_start_ns = time.monotonic_ns()
        block_reason = self._freshness_block_reason(command, now_ns=send_start_ns)
        if block_reason:
            return self._blocked_result(
                executor=executor,
                command=command,
                reason=block_reason,
                checked_ts_ns=send_start_ns,
            )

        generation = int(command.actuation_sequence or command.trajectory_generation or 0)
        self._last_generation = generation
        if command.action == "move" and command.dx == 0 and command.dy == 0:
            return ExecutionResult(
                executor_id=executor.executor_id,
                sent=False,
                intent=command,
                message="zero control command ignored",
                metadata={
                    **self._base_metadata(executor),
                    "action": "zero_output",
                    "command_generation": generation,
                    "freshness_checked_ts_ns": send_start_ns,
                },
            )
        try:
            result = executor.execute(command)
        except Exception as exc:
            return ExecutionResult(
                executor_id=executor.executor_id,
                sent=False,
                intent=command,
                message=f"mouse command send failed: {exc}",
                metadata={
                    **self._base_metadata(executor),
                    "action": "device_error",
                    "command_generation": generation,
                    "freshness_checked_ts_ns": send_start_ns,
                    "device_send_start_ts_ns": send_start_ns,
                    "device_send_end_ts_ns": time.monotonic_ns(),
                    "device_send_clock_domain": "monotonic",
                    "error": str(exc),
                },
            )
        send_end_ns = time.monotonic_ns()
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                **self._base_metadata(executor),
                "command_generation": generation,
                "freshness_checked_ts_ns": send_start_ns,
                "device_send_start_ts_ns": send_start_ns,
                "device_send_end_ts_ns": send_end_ns,
                "device_send_clock_domain": "monotonic",
            }
        )
        return replace(result, metadata=metadata)

    def _contract_block_reason(
        self,
        executor: Executor,
        command: ControlOutput,
    ) -> str:
        if not command.accepted:
            return "CONTROL_OUTPUT_REJECTED"
        if (
            not isinstance(command.dx, int)
            or isinstance(command.dx, bool)
            or not isinstance(command.dy, int)
            or isinstance(command.dy, bool)
        ):
            return "COMMAND_COUNTS_NOT_INTEGER"
        if (
            abs(command.dx) > self.MAX_ABS_DEVICE_COUNT
            or abs(command.dy) > self.MAX_ABS_DEVICE_COUNT
        ):
            return "COMMAND_COUNTS_OUT_OF_DEVICE_RANGE"
        if not isinstance(command.trigger_required, bool):
            return "TRIGGER_REQUIREMENT_MISSING"
        if command.trigger_required:
            if command.trigger_active is not True:
                return "TRIGGER_INACTIVE"
            reader = getattr(executor, "read_buttons", None)
            if not callable(reader):
                return "TRIGGER_STATE_UNAVAILABLE"
            try:
                state = reader()
            except Exception:
                return "TRIGGER_STATE_UNAVAILABLE"
            if not isinstance(state, dict) or state.get("available") is not True:
                return "TRIGGER_STATE_UNAVAILABLE"
            pressed = bool(state.get("left")) if command.left_trigger_required else bool(
                state.get("left") or state.get("right") or state.get("side")
            )
            if not pressed:
                return "TRIGGER_INACTIVE"
        return ""

    def _freshness_block_reason(
        self,
        command: ControlOutput,
        *,
        now_ns: int,
    ) -> str:
        expires_ts_ns = command.command_expires_ts_ns
        if expires_ts_ns is None or int(expires_ts_ns) <= 0:
            return "COMMAND_FRESHNESS_MISSING"
        if now_ns > int(expires_ts_ns):
            return "COMMAND_STALE"
        generation = command.actuation_sequence
        if generation is None:
            generation = command.trajectory_generation
        if generation is None:
            return "COMMAND_GENERATION_MISSING"
        if self._last_generation is not None and int(generation) <= self._last_generation:
            return "COMMAND_ACTUATION_NOT_NEWER"
        return ""

    def _blocked_result(
        self,
        *,
        executor: Executor,
        command: ControlOutput,
        reason: str,
        checked_ts_ns: int,
    ) -> ExecutionResult:
        return ExecutionResult(
            executor_id=executor.executor_id,
            sent=False,
            intent=command,
            message=f"mouse command blocked: {reason}",
            metadata={
                **self._base_metadata(executor),
                "action": "blocked",
                "block_reason": reason,
                "freshness_checked_ts_ns": checked_ts_ns,
            },
        )

    @staticmethod
    def _base_metadata(executor: Executor) -> dict[str, object]:
        return {
            "stage": "mouse_command_executor",
            "delivery_mode": "single_command_per_observation",
            "selected_executor": executor.executor_id,
            "scheduler_enabled": False,
        }
