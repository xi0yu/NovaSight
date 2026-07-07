from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import CommandScheduler, ControlOutputPolicy
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
        result = self.executors[self.selected].execute(bounded)
        scheduler_execution_metadata: dict[str, Any] | None = None
        scheduler_execution_metadata = self.scheduler.record_execution_result(
            sent=bool(result.sent),
            message=str(result.message),
        )
        if result.metadata is not None:
            if scheduler_metadata is not None or scheduler_execution_metadata is not None:
                metadata = dict(result.metadata)
                if scheduler_metadata is not None:
                    metadata["scheduler"] = {
                        **scheduler_metadata,
                        **({"execution": scheduler_execution_metadata} if scheduler_execution_metadata is not None else {}),
                    }
                elif scheduler_execution_metadata is not None:
                    metadata["scheduler"] = {"execution": scheduler_execution_metadata}
                return replace(result, metadata=metadata)
            return result
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
    return ControlOutputPolicy(
        max_abs_dx=config.control.max_abs_dx,
        max_abs_dy=config.control.max_abs_dy,
        min_confidence=config.control.min_confidence,
    )


def scheduler_from_config(config: RuntimeConfig) -> CommandScheduler:
    move_kind = str(getattr(config.control, "move_kind", "raw") or "raw")
    move_ms = max(0.0, float(getattr(config.control, "move_ms", 0)))
    movement_interval_ms = move_ms if move_kind in {"auto", "enc_auto", "bezier", "enc_bezier"} else 0.0
    interval_s = max(0.0, config.control.command_interval_ms, movement_interval_ms) / 1000.0
    return CommandScheduler(
        min_interval_s=interval_s,
        ttl_s=max(0.001, float(config.control.scheduler_command_ttl_ms) / 1000.0),
        predicted_ttl_s=max(0.001, float(config.control.scheduler_predicted_command_ttl_ms) / 1000.0),
        cancel_on_new_frame=bool(config.control.scheduler_cancel_on_new_frame),
        cancel_on_direction_change=bool(config.control.scheduler_cancel_on_direction_change),
        cancel_on_track_change=bool(config.control.scheduler_cancel_on_track_change),
        max_step_x=int(config.control.scheduler_max_step_x),
        max_step_y=int(config.control.scheduler_max_step_y),
        queue_hard_limit=int(config.control.scheduler_queue_hard_limit),
        device_error_cooldown_s=max(0.0, float(config.control.scheduler_device_error_cooldown_ms) / 1000.0),
    )
