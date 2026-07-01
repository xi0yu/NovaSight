from __future__ import annotations

from dataclasses import dataclass

from novasight.executors import ExecutionResult
from novasight.plugins import PluginBatchResult


@dataclass(frozen=True)
class RuntimeState:
    running: bool
    source: str
    active_model: dict | None
    executor: dict
    capture: dict
    inference: dict


@dataclass(frozen=True)
class RuntimeFrameResult:
    plugin_batch: PluginBatchResult
    execution_results: list[ExecutionResult]
