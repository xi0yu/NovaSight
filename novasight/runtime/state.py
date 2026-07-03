from __future__ import annotations

from dataclasses import dataclass

from novasight.executors import ExecutionResult
from novasight.contracts import ControlIntent


@dataclass(frozen=True)
class RuntimeState:
    running: bool
    source: str
    active_model: dict | None
    executor: dict
    capture: dict
    statistics: dict
    inference: dict
    config: dict
    pipeline: dict
    vision: dict
    fatal_error: dict | None


@dataclass(frozen=True)
class RuntimeFrameResult:
    control_intents: list[ControlIntent]
    execution_results: list[ExecutionResult]
