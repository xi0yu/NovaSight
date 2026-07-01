from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from novasight.plugins import ControlIntent


@dataclass(frozen=True)
class ExecutionResult:
    executor_id: str
    sent: bool
    intent: ControlIntent
    message: str = ""


class Executor(Protocol):
    executor_id: str

    def available(self) -> bool:
        ...

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        ...
