from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import time
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import (
    MAX_PLAN_DURATION_MS,
    CommandScheduler,
    ControlOutput,
    ControlOutputPolicy,
    plan_step_capacity,
)
from novasight.executors.contracts import ExecutionResult, Executor
from novasight.executors.kmnet import KmNetExecutor
from novasight.contracts import ControlIntent


class ExecutorRegistry:
    def __init__(
        self,
        executors: Iterable[Executor],
        default: str = "kmnet",
        policy: ControlOutputPolicy | None = None,
        scheduler: CommandScheduler | None = None,
    ) -> None:
        self.executors = {executor.executor_id: executor for executor in executors}
        if default not in self.executors:
            raise ValueError(f"unknown executor: {default}")
        self.selected = default
        self.policy = policy or ControlOutputPolicy()
        self.scheduler = scheduler

    @classmethod
    def with_builtin_executors(
        cls,
        config: RuntimeConfig | None = None,
        default: str = "kmnet",
        policy: ControlOutputPolicy | None = None,
        scheduler: CommandScheduler | None = None,
    ) -> ExecutorRegistry:
        kmnet = KmNetExecutor.from_config(config) if config is not None else KmNetExecutor()
        return cls(
            executors=[
                kmnet,
            ],
            default=default,
            policy=policy,
            scheduler=scheduler,
        )

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> ExecutorRegistry:
        return cls.with_builtin_executors(
            config=config,
            default=config.control.output_mode or config.executor.default,
            policy=policy_from_config(config),
            scheduler=scheduler_from_config(config),
        )

    def update_runtime_config(self, config: RuntimeConfig) -> None:
        selected = config.control.output_mode or config.executor.default
        if selected not in self.executors:
            raise ValueError(f"unknown executor: {selected}")
        self.selected = selected
        self.policy = policy_from_config(config)
        self.scheduler = scheduler_from_config(config)

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        bounded = self.policy.apply(intent)
        if self.scheduler is None:
            return ExecutionResult(
                executor_id=self.selected,
                sent=False,
                intent=bounded,
                message="command scheduler required",
                metadata={
                    "stage": "scheduler_required",
                    "selected_executor": self.selected,
                    "accepted": bool(bounded.accepted),
                    "clipped": bool(bounded.clipped),
                    "policy_reason": str(bounded.reason),
                },
            )
        scheduler_metadata: dict[str, Any] | None = None
        decision = self.scheduler.submit(bounded)
        scheduler_metadata = decision.metadata
        if decision.output is None:
            return ExecutionResult(
                executor_id=self.selected,
                sent=False,
                intent=bounded,
                message="control command scheduled",
                metadata={
                    "stage": "scheduler",
                    "selected_executor": self.selected,
                    **scheduler_metadata,
                },
            )
        bounded = decision.output
        device_send_start_ts_ns = time.monotonic_ns()
        result = self.executors[self.selected].execute(bounded)
        device_send_end_ts_ns = time.monotonic_ns()
        scheduler_execution_metadata: dict[str, Any] | None = None
        scheduler_execution_metadata = self.scheduler.record_execution_result(
            sent=bool(result.sent),
            message=str(result.message),
        )
        device_timing_metadata = {
            "device_send_start_ts_ns": device_send_start_ts_ns,
            "device_send_end_ts_ns": device_send_end_ts_ns,
            "device_send_clock_domain": "monotonic",
            "scheduler_send_delay_us": _scheduler_send_delay_us(
                scheduler_metadata,
                device_send_start_ts_ns,
            ),
        }
        if result.metadata is not None:
            if scheduler_metadata is not None or scheduler_execution_metadata is not None:
                metadata = dict(result.metadata)
                metadata.update(device_timing_metadata)
                if scheduler_metadata is not None:
                    metadata["scheduler"] = {
                        **scheduler_metadata,
                        **({"execution": scheduler_execution_metadata} if scheduler_execution_metadata is not None else {}),
                    }
                elif scheduler_execution_metadata is not None:
                    metadata["scheduler"] = {"execution": scheduler_execution_metadata}
                return replace(result, metadata=metadata)
            metadata = dict(result.metadata)
            metadata.update(device_timing_metadata)
            return replace(result, metadata=metadata)
        return ExecutionResult(
            executor_id=result.executor_id,
            sent=result.sent,
            intent=result.intent,
            message=result.message,
            metadata={
                "stage": "executor",
                "selected_executor": self.selected,
                "accepted": bool(bounded.accepted),
                "clipped": bool(bounded.clipped),
                "policy_reason": str(bounded.reason),
                **device_timing_metadata,
                **(
                    {
                        "scheduler": {
                            **(scheduler_metadata or {}),
                            **({"execution": scheduler_execution_metadata} if scheduler_execution_metadata is not None else {}),
                        }
                    }
                    if scheduler_metadata is not None or scheduler_execution_metadata is not None
                    else {}
                ),
            },
        )

    def tick_pending(self, *, now_s: float | None = None) -> ExecutionResult:
        if self.scheduler is None:
            return ExecutionResult(
                executor_id=self.selected,
                sent=False,
                intent=_scheduler_status_output("scheduler_required"),
                message="command scheduler required",
                metadata={
                    "stage": "scheduler_required",
                    "selected_executor": self.selected,
                },
            )
        decision = self.scheduler.tick(now_s=now_s)
        scheduler_metadata = decision.metadata
        if decision.output is None:
            return ExecutionResult(
                executor_id=self.selected,
                sent=False,
                intent=_scheduler_status_output(str(scheduler_metadata.get("action") or "scheduler_idle")),
                message="no pending control command ready",
                metadata={
                    "stage": "scheduler",
                    "selected_executor": self.selected,
                    **scheduler_metadata,
                },
            )
        device_send_start_ts_ns = time.monotonic_ns()
        result = self.executors[self.selected].execute(decision.output)
        device_send_end_ts_ns = time.monotonic_ns()
        scheduler_execution_metadata = self.scheduler.record_execution_result(
            sent=bool(result.sent),
            message=str(result.message),
            now_s=now_s,
        )
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "device_send_start_ts_ns": device_send_start_ts_ns,
                "device_send_end_ts_ns": device_send_end_ts_ns,
                "device_send_clock_domain": "monotonic",
                "scheduler_send_delay_us": _scheduler_send_delay_us(
                    scheduler_metadata,
                    device_send_start_ts_ns,
                ),
            }
        )
        metadata["scheduler"] = {
            **scheduler_metadata,
            "execution": scheduler_execution_metadata,
        }
        return replace(result, metadata=metadata)

    def status(self, *, refresh_buttons: bool = False) -> dict[str, Any]:
        def executor_status(executor: Executor) -> dict[str, Any]:
            status = getattr(executor, "status", None)
            if not callable(status):
                return {"available": executor.available()}
            try:
                return status(refresh_buttons=refresh_buttons)
            except TypeError:
                return status()

        return {
            "selected": self.selected,
            "scheduler": self.scheduler.status() if self.scheduler is not None else {"enabled": False},
            "executors": {
                executor_id: executor_status(executor)
                for executor_id, executor in self.executors.items()
            },
        }

    def read_buttons(self) -> dict[str, Any]:
        kmnet = self.executors.get("kmnet")
        reader = getattr(kmnet, "read_buttons", None)
        if not callable(reader):
            return {"available": False, "left": False, "right": False, "reason": "kmNet executor is unavailable"}
        return reader()


def policy_from_config(config: RuntimeConfig) -> ControlOutputPolicy:
    capacity = plan_step_capacity(config.control.scheduler_interval_ms)
    return ControlOutputPolicy(
        max_abs_dx=int(config.control.scheduler_step_counts_x) * capacity,
        max_abs_dy=int(config.control.scheduler_step_counts_y) * capacity,
        min_confidence=config.control.min_confidence,
    )


def scheduler_from_config(config: RuntimeConfig) -> CommandScheduler:
    interval_ms = max(1.0, min(10.0, float(config.control.scheduler_interval_ms)))
    interval_s = interval_ms / 1000.0
    capacity = plan_step_capacity(config.control.scheduler_interval_ms)
    expiry_s = (MAX_PLAN_DURATION_MS + interval_ms) / 1000.0
    return CommandScheduler(
        min_interval_s=interval_s,
        ttl_s=expiry_s,
        predicted_ttl_s=expiry_s,
        cancel_on_new_frame=True,
        cancel_on_direction_change=True,
        cancel_on_track_change=True,
        max_step_x=int(config.control.scheduler_step_counts_x),
        max_step_y=int(config.control.scheduler_step_counts_y),
        queue_hard_limit=capacity,
        device_error_cooldown_s=0.050,
    )


def _scheduler_status_output(reason: str) -> ControlOutput:
    return ControlOutput(
        dx=0,
        dy=0,
        action=None,
        confidence=0.0,
        source_id="scheduler",
        accepted=False,
        clipped=False,
        reason=reason,
    )


def _scheduler_send_delay_us(metadata: dict[str, Any], send_start_ts_ns: int) -> float | None:
    scheduled_ts_ns = metadata.get("scheduled_ts_ns")
    if not isinstance(scheduled_ts_ns, (int, float)) or isinstance(scheduled_ts_ns, bool):
        return None
    return max(0.0, (int(send_start_ts_ns) - int(scheduled_ts_ns)) / 1000.0)
