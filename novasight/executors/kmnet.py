from __future__ import annotations

from typing import Any

from novasight.executors.contracts import ExecutionResult
from novasight.plugins import ControlIntent


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

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        if self._driver is None:
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=False,
                intent=intent,
                message="kmNet driver unavailable",
            )

        self._driver.move(int(round(intent.dx)), int(round(-intent.dy)))
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=True,
            intent=intent,
            message="sent intent to kmNet driver",
        )
