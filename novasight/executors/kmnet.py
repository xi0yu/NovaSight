from __future__ import annotations

from types import ModuleType

from novasight.executors.contracts import ExecutionResult
from novasight.plugins import ControlIntent


class KmNetExecutor:
    executor_id = "kmnet"

    def __init__(self) -> None:
        try:
            import kmNet
        except ImportError:
            self._driver: ModuleType | None = None
        else:
            self._driver = kmNet

    @property
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
