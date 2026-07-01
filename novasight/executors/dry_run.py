from __future__ import annotations

from novasight.executors.contracts import ExecutionResult
from novasight.plugins import ControlIntent


class DryRunExecutor:
    executor_id = "dry_run"

    def __init__(self) -> None:
        self.history: list[ControlIntent] = []

    def available(self) -> bool:
        return True

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        self.history.append(intent)
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=False,
            intent=intent,
            message="recorded intent without external action",
        )
