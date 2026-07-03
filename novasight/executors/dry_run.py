from __future__ import annotations

import sys

from novasight.control import ControlOutput
from novasight.executors.contracts import ExecutionResult


class DryRunExecutor:
    executor_id = "dry_run"

    def __init__(self) -> None:
        self.history: list[ControlOutput] = []

    def available(self) -> bool:
        return True

    def execute(self, output: ControlOutput) -> ExecutionResult:
        self.history.append(output)
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=False,
            intent=output,
            message="recorded output without external action",
        )


class SilentExecutor:
    executor_id = "silent"

    def __init__(self) -> None:
        self.history: list[ControlOutput] = []

    def available(self) -> bool:
        return True

    def execute(self, output: ControlOutput) -> ExecutionResult:
        self.history.append(output)
        return ExecutionResult("silent", False, output, "swallowed")


class ConsoleExecutor:
    executor_id = "console"

    def available(self) -> bool:
        return True

    def execute(self, output: ControlOutput) -> ExecutionResult:
        print(
            "control "
            f"dx={output.dx} "
            f"dy={output.dy} "
            f"action={output.action} "
            f"confidence={output.confidence:.2f} "
            f"source={output.source_id} "
            f"accepted={output.accepted}",
            file=sys.stdout,
        )
        return ExecutionResult("console", False, output, "printed")
