from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from novasight.control import ControlOutput


@dataclass(frozen=True)
class ExecutionResult:
    executor_id: str
    sent: bool
    intent: ControlOutput
    message: str = ""
    metadata: dict | None = None


@dataclass(frozen=True)
class BoxInputState:
    left: bool = False
    right: bool = False
    side: bool = False
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.left or self.right or self.side


class Executor(Protocol):
    executor_id: str

    def available(self) -> bool:
        ...

    def execute(self, output: ControlOutput) -> ExecutionResult:
        ...
