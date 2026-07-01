from __future__ import annotations

from typing import Any

from novasight.control import ControlOutput
from novasight.executors.contracts import ExecutionResult


class KmNetExecutor:
    executor_id = "kmnet"

    def __init__(self) -> None:
        try:
            import kmNet
        except Exception:
            self._driver: Any | None = None
        else:
            self._driver = kmNet

    def available(self) -> bool:
        return self._driver is not None

    def execute(self, output: ControlOutput) -> ExecutionResult:
        if not output.accepted:
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=False,
                intent=output,
                message="control output rejected",
            )
        if self._driver is None:
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=False,
                intent=output,
                message="kmNet driver unavailable",
            )

        self._driver.move(output.dx, -output.dy)
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=True,
            intent=output,
            message="sent output to kmNet driver",
        )
