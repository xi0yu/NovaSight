from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import time
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import ControlCommandCoalescer, ControlOutput, ControlOutputPolicy
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
        y_limiter: YAxisWindowLimiter | None = None,
    ) -> None:
        self.executors = {executor.executor_id: executor for executor in executors}
        if default not in self.executors:
            raise ValueError(f"unknown executor: {default}")
        self.selected = default
        self.policy = policy or ControlOutputPolicy()
        self.coalescer = coalescer
        self.y_limiter = y_limiter

    @classmethod
    def with_builtin_executors(
        cls,
        config: RuntimeConfig | None = None,
        default: str = "dry_run",
        policy: ControlOutputPolicy | None = None,
        coalescer: ControlCommandCoalescer | None = None,
        y_limiter: YAxisWindowLimiter | None = None,
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
            y_limiter=y_limiter,
        )

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> ExecutorRegistry:
        return cls.with_builtin_executors(
            config=config,
            default=config.control.output_mode or config.executor.default,
            policy=policy_from_config(config),
            coalescer=coalescer_from_config(config),
            y_limiter=YAxisWindowLimiter.from_config(config),
        )

    def update_runtime_config(self, config: RuntimeConfig) -> None:
        selected = config.control.output_mode or config.executor.default
        if selected not in self.executors:
            raise ValueError(f"unknown executor: {selected}")
        self.selected = selected
        self.policy = policy_from_config(config)
        self.coalescer = coalescer_from_config(config)
        if self.y_limiter is None:
            self.y_limiter = YAxisWindowLimiter.from_config(config)
        else:
            self.y_limiter.configure_from_config(config)

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
                    metadata={
                        "stage": "coalescer",
                        "selected_executor": self.selected,
                        "requested_dx": float(intent.dx),
                        "requested_dy": float(intent.dy),
                        "pending_dx": float(getattr(getattr(self.coalescer, "_pending_intent", None), "dx", 0.0)),
                        "pending_dy": float(getattr(getattr(self.coalescer, "_pending_intent", None), "dy", 0.0)),
                        "dropped_since_emit": int(getattr(self.coalescer, "_dropped_since_emit", 0)),
                        "min_interval_ms": float(self.coalescer.min_interval_s * 1000.0),
                    },
                )
            intent = merged
        bounded = self.policy.apply(intent)
        limiter_metadata: dict[str, Any] | None = None
        if self.y_limiter is not None:
            bounded, limiter_metadata = self.y_limiter.apply(bounded)
        result = self.executors[self.selected].execute(bounded)
        if result.metadata is not None:
            if limiter_metadata is not None:
                metadata = dict(result.metadata)
                metadata["y_rate_limiter"] = limiter_metadata
                return replace(result, metadata=metadata)
            return result
        return ExecutionResult(
            executor_id=result.executor_id,
            sent=result.sent,
            intent=result.intent,
            message=result.message,
            metadata={
                "stage": "executor",
                "selected_executor": self.selected,
                "accepted": bool(bounded.accepted),
                "clipped": bool(bounded.clipped),
                "policy_reason": str(bounded.reason),
                **({"y_rate_limiter": limiter_metadata} if limiter_metadata is not None else {}),
            },
        )

    def status(self, *, refresh_buttons: bool = False) -> dict[str, Any]:
        def executor_status(executor: Executor) -> dict[str, Any]:
            status = getattr(executor, "status", None)
            if not callable(status):
                return {"available": executor.available()}
            try:
                return status(refresh_buttons=refresh_buttons)
            except TypeError:
                return status()

        return {
            "selected": self.selected,
            "executors": {
                executor_id: executor_status(executor)
                for executor_id, executor in self.executors.items()
            },
        }

    def read_buttons(self) -> dict[str, Any]:
        kmnet = self.executors.get("kmnet")
        reader = getattr(kmnet, "read_buttons", None)
        if not callable(reader):
            return {"available": False, "left": False, "right": False, "reason": "kmNet executor is unavailable"}
        return reader()


class YAxisWindowLimiter:
    @classmethod
    def from_config(cls, config: RuntimeConfig) -> YAxisWindowLimiter:
        return cls(
            enabled=bool(config.control.y_down_enabled)
            and config.control.strategy not in {"isolated_mouse", "dynamic_pid"},
            window_s=max(0.0, config.control.y_rate_window_ms / 1000.0),
            max_counts=max(0.0, config.control.y_rate_max_counts),
        )

    def __init__(self, *, enabled: bool, window_s: float, max_counts: float) -> None:
        self.enabled = bool(enabled)
        self.window_s = max(0.0, window_s)
        self.max_counts = max(0.0, max_counts)
        self._last_drop_s = time.monotonic() if self.enabled else 0.0

    def configure_from_config(self, config: RuntimeConfig) -> None:
        next_enabled = bool(config.control.y_down_enabled)
        if config.control.strategy in {"isolated_mouse", "dynamic_pid"}:
            next_enabled = False
        was_enabled = self.enabled
        self.enabled = next_enabled
        self.window_s = max(0.0, config.control.y_rate_window_ms / 1000.0)
        self.max_counts = max(0.0, config.control.y_rate_max_counts)
        if next_enabled and not was_enabled:
            self._last_drop_s = time.monotonic()
        elif not next_enabled:
            self._last_drop_s = 0.0

    def apply(self, output: ControlOutput, now_s: float | None = None) -> tuple[ControlOutput, dict[str, Any]]:
        now = time.monotonic() if now_s is None else now_s
        if not self.enabled or self.window_s <= 0 or self.max_counts <= 0 or not output.accepted:
            return output, {
                "enabled": False,
                "window_ms": self.window_s * 1000.0,
                "max_counts": self.max_counts,
                "applied": False,
                "drop_counts": 0,
            }
        elapsed_s = now - self._last_drop_s if self._last_drop_s > 0 else self.window_s
        should_drop = self._last_drop_s <= 0 or elapsed_s >= self.window_s
        drop_counts = -int(round(self.max_counts)) if should_drop else 0
        if should_drop:
            self._last_drop_s = now
        requested_dy = int(round(output.dy))
        final_dy = requested_dy + drop_counts
        reason = "Y axis periodic down compensation" if should_drop else output.reason
        return replace(output, dy=final_dy, reason=reason), {
            "enabled": True,
            "window_ms": self.window_s * 1000.0,
            "max_counts": self.max_counts,
            "requested_dy": requested_dy,
            "drop_counts": drop_counts,
            "final_dy": final_dy,
            "applied": should_drop,
            "elapsed_ms": elapsed_s * 1000.0,
        }


def policy_from_config(config: RuntimeConfig) -> ControlOutputPolicy:
    return ControlOutputPolicy(
        max_abs_dx=config.control.max_abs_dx,
        max_abs_dy=config.control.max_abs_dy,
        min_confidence=config.control.min_confidence,
    )


def coalescer_from_config(config: RuntimeConfig) -> ControlCommandCoalescer:
    move_kind = str(getattr(config.control, "move_kind", "raw") or "raw")
    move_ms = max(0.0, float(getattr(config.control, "move_ms", 0)))
    movement_interval_ms = move_ms if move_kind in {"auto", "enc_auto", "bezier", "enc_bezier"} else 0.0
    return ControlCommandCoalescer(
        min_interval_s=max(0.0, config.control.command_interval_ms, movement_interval_ms) / 1000.0
    )
