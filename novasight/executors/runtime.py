from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import ControlCommandCoalescer, ControlOutputPolicy
from novasight.executors.contracts import ExecutionResult, Executor
from novasight.executors.dry_run import ConsoleExecutor, DryRunExecutor, SilentExecutor
from novasight.executors.kmnet import KmNetExecutor
from novasight.contracts import ControlIntent


class ExecutorRegistry:
    def __init__(
        self,
        executors: Iterable[Executor],
        default: str = "dry_run",
        policy: ControlOutputPolicy | None = None,
        coalescer: ControlCommandCoalescer | None = None,
    ) -> None:
        self.executors = {executor.executor_id: executor for executor in executors}
        if default not in self.executors:
            raise ValueError(f"unknown executor: {default}")
        self.selected = default
        self.policy = policy or ControlOutputPolicy()
        self.coalescer = coalescer

    @classmethod
    def with_builtin_executors(
        cls,
        config: RuntimeConfig | None = None,
        default: str = "dry_run",
        policy: ControlOutputPolicy | None = None,
        coalescer: ControlCommandCoalescer | None = None,
    ) -> ExecutorRegistry:
        kmnet = KmNetExecutor.from_config(config) if config is not None else KmNetExecutor()
        return cls(
            executors=[
                SilentExecutor(),
                ConsoleExecutor(),
                DryRunExecutor(),
                kmnet,
            ],
            default=default,
            policy=policy,
            coalescer=coalescer,
        )

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> ExecutorRegistry:
        default = config.control.output_mode or config.executor.default
        return cls.with_builtin_executors(
            config=config,
            default=default,
            policy=ControlOutputPolicy(
                max_abs_dx=config.control.max_abs_dx,
                max_abs_dy=config.control.max_abs_dy,
                min_confidence=config.control.min_confidence,
            ),
            coalescer=ControlCommandCoalescer(
                min_interval_s=max(0.0, config.control.command_interval_ms / 1000.0)
            ),
        )

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        if self.coalescer is not None:
            merged = self.coalescer.push(intent)
            if merged is None:
                bounded = self.policy.apply(intent)
                return ExecutionResult(
                    executor_id=self.selected,
                    sent=False,
                    intent=bounded,
                    message="control command coalesced",
                    metadata={
                        "stage": "coalescer",
                        "selected_executor": self.selected,
                        "requested_dx": float(intent.dx),
                        "requested_dy": float(intent.dy),
                        "pending_dx": float(getattr(self.coalescer, "_pending_dx", 0.0)),
                        "pending_dy": float(getattr(self.coalescer, "_pending_dy", 0.0)),
                        "min_interval_ms": float(self.coalescer.min_interval_s * 1000.0),
                    },
                )
            intent = merged
        bounded = self.policy.apply(intent)
        result = self.executors[self.selected].execute(bounded)
        if result.metadata is not None:
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
