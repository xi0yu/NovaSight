from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from novasight.executors.contracts import ExecutionResult, Executor
from novasight.executors.dry_run import DryRunExecutor
from novasight.executors.kmnet import KmNetExecutor
from novasight.plugins import ControlIntent


class ExecutorRegistry:
    def __init__(
        self,
        executors: Iterable[Executor],
        default: str = "dry_run",
    ) -> None:
        self.executors = {executor.executor_id: executor for executor in executors}
        if default not in self.executors:
            raise ValueError(f"unknown executor: {default}")
        self.selected = default

    @classmethod
    def with_builtin_executors(cls, default: str = "dry_run") -> ExecutorRegistry:
        return cls(
            executors=[DryRunExecutor(), KmNetExecutor()],
            default=default,
        )

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        return self.executors[self.selected].execute(intent)

    def status(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "executors": {
                executor_id: {"available": executor.available}
                for executor_id, executor in self.executors.items()
            },
        }
