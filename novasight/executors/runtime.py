from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import ControlOutputPolicy
from novasight.executors.contracts import ExecutionResult, Executor
from novasight.executors.dry_run import ConsoleExecutor, DryRunExecutor, SilentExecutor
from novasight.executors.kmnet import KmNetExecutor
from novasight.plugins import ControlIntent


class ExecutorRegistry:
    def __init__(
        self,
        executors: Iterable[Executor],
        default: str = "dry_run",
        policy: ControlOutputPolicy | None = None,
    ) -> None:
        self.executors = {executor.executor_id: executor for executor in executors}
        if default not in self.executors:
            raise ValueError(f"unknown executor: {default}")
        self.selected = default
        self.policy = policy or ControlOutputPolicy()

    @classmethod
    def with_builtin_executors(
        cls,
        default: str = "dry_run",
        policy: ControlOutputPolicy | None = None,
    ) -> ExecutorRegistry:
        return cls(
            executors=[
                SilentExecutor(),
                ConsoleExecutor(),
                DryRunExecutor(),
                KmNetExecutor(),
            ],
            default=default,
            policy=policy,
        )

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> ExecutorRegistry:
        return cls.with_builtin_executors(
            default=config.control.output_mode,
            policy=ControlOutputPolicy(
                max_abs_dx=config.control.max_abs_dx,
                max_abs_dy=config.control.max_abs_dy,
                min_confidence=config.control.min_confidence,
            ),
        )

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        bounded = self.policy.apply(intent)
        return self.executors[self.selected].execute(bounded)

    def status(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "executors": {
                executor_id: {"available": executor.available()}
                for executor_id, executor in self.executors.items()
            },
        }
