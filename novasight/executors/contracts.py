from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from novasight.control import ControlOutput


@dataclass(frozen=True)
class ExecutionResult:
    executor_id: str
    sent: bool
    intent: ControlOutput
    message: str = ""


class Executor(Protocol):
    executor_id: str

    def available(self) -> bool:
        ...

    def execute(self, output: ControlOutput) -> ExecutionResult:
        ...
