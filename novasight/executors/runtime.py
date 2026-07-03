from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import ControlCommandCoalescer, ControlOutputPolicy
from novasight.executors.contracts import ExecutionResult, Executor
from novasight.executors.dry_run import ConsoleExecutor, DryRunExecutor, SilentExecutor
from novasight.executors.kmnet import KmNetExecutor
from novasight.contracts import ControlIntent


class ExecutorRegistry:
    def __init__(
        self,
        executors: Iterable[Executor],
        default: str = "dry_run",
        policy: ControlOutputPolicy | None = None,
        coalescer: ControlCommandCoalescer | None = None,
    ) -> None:
        self.executors = {executor.executor_id: executor for executor in executors}
        if default not in self.executors:
            raise ValueError(f"unknown executor: {default}")
        self.selected = default
        self.policy = policy or ControlOutputPolicy()
        self.coalescer = coalescer

    @classmethod
    def with_builtin_executors(
        cls,
        config: RuntimeConfig | None = None,
        default: str = "dry_run",
        policy: ControlOutputPolicy | None = None,
        coalescer: ControlCommandCoalescer | None = None,
    ) -> ExecutorRegistry:
        kmnet = KmNetExecutor.from_config(config) if config is not None else KmNetExecutor()
        return cls(
            executors=[
                SilentExecutor(),
                ConsoleExecutor(),
                DryRunExecutor(),
                kmnet,
            ],
            default=default,
            policy=policy,
            coalescer=coalescer,
        )

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> ExecutorRegistry:
        default = config.control.output_mode or config.executor.default
        return cls.with_builtin_executors(
            config=config,
            default=default,
            policy=ControlOutputPolicy(
                max_abs_dx=config.control.max_abs_dx,
                max_abs_dy=config.control.max_abs_dy,
                min_confidence=config.control.min_confidence,
            ),
            coalescer=ControlCommandCoalescer(
                min_interval_s=max(0.0, config.control.command_interval_ms / 1000.0)
            ),
        )

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        if self.coalescer is not None:
            merged = self.coalescer.push(intent)
            if merged is None:
                bounded = self.policy.apply(intent)
                return ExecutionResult(
                    executor_id=self.selected,
                    sent=False,
                    intent=bounded,
                    message="control command coalesced",
                )
            intent = merged
        bounded = self.policy.apply(intent)
        return self.executors[self.selected].execute(bounded)

    def status(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "executors": {
                executor_id: (
                    executor.status()
                    if hasattr(executor, "status") and callable(getattr(executor, "status"))
                    else {"available": executor.available()}
                )
                for executor_id, executor in self.executors.items()
            },
        }
