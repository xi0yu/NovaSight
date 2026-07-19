from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import math
import threading
import time
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import (
    ControlMixer,
    DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2,
    MAX_PLAN_DURATION_MS,
    CommandScheduler,
    ControlOutput,
    ControlOutputPolicy,
    plan_step_capacity,
)
from novasight.control.recoil import RecoilDecision, RecoilState
from novasight.control.registry import SchedulerPolicy, algorithm_definition
from novasight.executors.contracts import ExecutionResult, Executor
from novasight.executors.kmnet import KmNetExecutor
from novasight.executors.mouse_command import MouseCommandExecutor
from novasight.contracts import ControlIntent


class ExecutorRegistry:
    def __init__(
        self,
        executors: Iterable[Executor],
        default: str = "kmnet",
        policy: ControlOutputPolicy | None = None,
        scheduler: CommandScheduler | None = None,
        direct_output: bool = False,
        single_command_per_observation: bool = False,
        latest_replace: bool = False,
        output_enabled: bool = True,
    ) -> None:
        self.executors = {executor.executor_id: executor for executor in executors}
        if default not in self.executors:
            raise ValueError(f"unknown executor: {default}")
        self.selected = default
        self.policy = policy or ControlOutputPolicy()
        self.single_command_per_observation = bool(single_command_per_observation)
        self.latest_replace = bool(latest_replace)
        self.output_enabled = bool(output_enabled)
        self.scheduler = None if self.single_command_per_observation else scheduler
        self.direct_output = bool(direct_output) or self.single_command_per_observation
        self._config_epoch = 0
        self._submission_epoch = 0
        self._actuation_sequence = 0
        self._scheduler_lock = threading.Lock()
        self._executor_lock = threading.Lock()
        self.mouse_command_executor = MouseCommandExecutor(self._executor_lock)

    @classmethod
    def with_builtin_executors(
        cls,
        config: RuntimeConfig | None = None,
        default: str = "kmnet",
        policy: ControlOutputPolicy | None = None,
        scheduler: CommandScheduler | None = None,
        direct_output: bool = False,
        single_command_per_observation: bool = False,
        latest_replace: bool = False,
        output_enabled: bool = True,
    ) -> ExecutorRegistry:
        kmnet = KmNetExecutor.from_config(config) if config is not None else KmNetExecutor()
        return cls(
            executors=[
                kmnet,
            ],
            default=default,
            policy=policy,
            scheduler=scheduler,
            direct_output=direct_output,
            single_command_per_observation=single_command_per_observation,
            latest_replace=latest_replace,
            output_enabled=output_enabled,
        )

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> ExecutorRegistry:
        capabilities = algorithm_definition(str(config.control.active_algorithm)).capabilities
        latest_replace = bool(
            capabilities.scheduler_policy is SchedulerPolicy.LATEST_REPLACE
            or config.control.recoil.enabled
        )
        return cls.with_builtin_executors(
            config=config,
            default="kmnet",
            policy=policy_from_config(config),
            scheduler=scheduler_from_config(config),
            direct_output=not latest_replace and not bool(config.control.scheduler_enabled),
            latest_replace=latest_replace,
            output_enabled=bool(config.control.output_enabled),
        )

    def update_runtime_config(self, config: RuntimeConfig) -> None:
        selected = "kmnet"
        if selected not in self.executors:
            raise ValueError(f"unknown executor: {selected}")
        capabilities = algorithm_definition(str(config.control.active_algorithm)).capabilities
        latest_replace = bool(
            capabilities.scheduler_policy is SchedulerPolicy.LATEST_REPLACE
            or config.control.recoil.enabled
        )
        single_command = False
        scheduler = scheduler_from_config(config)
        direct_output = not latest_replace and not bool(config.control.scheduler_enabled)
        policy = policy_from_config(config)
        with self._scheduler_lock:
            if self.output_enabled and not bool(config.control.output_enabled):
                self._submission_epoch += 1
                if self.scheduler is not None:
                    self.scheduler.clear("CONTROL_OUTPUT_DISABLED")
            self.selected = selected
            self.policy = policy
            self.scheduler = scheduler
            self.direct_output = direct_output
            self.single_command_per_observation = single_command
            self.latest_replace = latest_replace
            self.output_enabled = bool(config.control.output_enabled)
            self._config_epoch += 1
            kmnet = self.executors.get("kmnet")
            if isinstance(kmnet, KmNetExecutor):
                kmnet.button_poll_interval_s = max(
                    0.001,
                    min(0.050, float(config.control.scheduler_interval_ms) / 1000.0),
                )

    def set_output_enabled(self, enabled: bool) -> None:
        """Atomically gate delivery without rebuilding scheduler or device state."""

        next_enabled = bool(enabled)
        with self._scheduler_lock:
            if self.output_enabled == next_enabled:
                return
            # Invalidate a command already popped by another thread.  Closing
            # the gate also drops every queued step so reopening can only
            # deliver an observation submitted after the transition.
            self._config_epoch += 1
            self._submission_epoch += 1
            if not next_enabled and self.scheduler is not None:
                self.scheduler.clear("CONTROL_OUTPUT_DISABLED")
            self.output_enabled = next_enabled

    def execute(self, intent: ControlIntent) -> ExecutionResult:
        with self._scheduler_lock:
            bounded = self.policy.apply(intent)
            single_command = self.single_command_per_observation
            latest_replace = self.latest_replace
            scheduler = None if single_command else self.scheduler
            direct_output = self.direct_output or single_command
            selected = self.selected
            config_epoch = self._config_epoch
            if not self.output_enabled:
                return _output_disabled_result(selected=selected, output=bounded)
            if scheduler is not None:
                self._submission_epoch += 1
                decision = scheduler.submit(bounded, emit_immediately=False)
            else:
                decision = None
        if scheduler is None:
            if direct_output:
                return self._execute_direct(
                    bounded,
                    expected_config_epoch=config_epoch,
                    expected_selected=selected,
                    expected_single_command=single_command,
                )
            return ExecutionResult(
                executor_id=selected,
                sent=False,
                intent=bounded,
                message="command scheduler required",
                metadata={
                    "stage": "scheduler_required",
                    "selected_executor": selected,
                    "accepted": bool(bounded.accepted),
                    "clipped": bool(bounded.clipped),
                    "policy_reason": str(bounded.reason),
                },
            )
        assert decision is not None
        scheduler_metadata: dict[str, Any] | None = None
        scheduler_metadata = decision.metadata
        if decision.output is None:
            message = (
                "zero control command ignored"
                if scheduler_metadata.get("action") == "zero_output"
                else "control command scheduled"
            )
            return ExecutionResult(
                executor_id=selected,
                sent=False,
                intent=bounded,
                message=message,
                metadata={
                    "stage": "scheduler",
                    "selected_executor": selected,
                    "delivery_mode": "latest_replace" if latest_replace else "scheduler",
                    "scheduler_enabled": True,
                    **scheduler_metadata,
                },
            )
        if latest_replace:
            return ExecutionResult(
                executor_id=selected,
                sent=False,
                intent=decision.output,
                message="control command rejected before latest-replace delivery",
                metadata={
                    "stage": "scheduler",
                    "selected_executor": selected,
                    "delivery_mode": "latest_replace",
                    "scheduler_enabled": True,
                    **scheduler_metadata,
                },
            )
        bounded = decision.output
        with self._executor_lock:
            with self._scheduler_lock:
                if (
                    self._config_epoch != config_epoch
                    or self.selected != selected
                    or self.single_command_per_observation
                    or self.scheduler is not scheduler
                ):
                    return _scheduler_superseded_result(
                        selected=selected,
                        output=bounded,
                    )
                device_send_start_ts_ns = time.monotonic_ns()
                result = self.executors[selected].execute(bounded)
                device_send_end_ts_ns = time.monotonic_ns()
                scheduler_execution_metadata = scheduler.record_execution_result(
                    sent=bool(result.sent),
                    message=str(result.message),
                )
        device_timing_metadata = {
            "device_send_start_ts_ns": device_send_start_ts_ns,
            "device_send_end_ts_ns": device_send_end_ts_ns,
            "device_send_clock_domain": "monotonic",
            "scheduler_send_delay_us": _scheduler_send_delay_us(
                scheduler_metadata,
                device_send_start_ts_ns,
            ),
        }
        if result.metadata is not None:
            if scheduler_metadata is not None or scheduler_execution_metadata is not None:
                metadata = dict(result.metadata)
                metadata.update(device_timing_metadata)
                if scheduler_metadata is not None:
                    metadata["scheduler"] = {
                        **scheduler_metadata,
                        **(
                            {"execution": scheduler_execution_metadata}
                            if scheduler_execution_metadata is not None
                            else {}
                        ),
                    }
                elif scheduler_execution_metadata is not None:
                    metadata["scheduler"] = {"execution": scheduler_execution_metadata}
                return replace(result, metadata=metadata)
            metadata = dict(result.metadata)
            metadata.update(device_timing_metadata)
            return replace(result, metadata=metadata)
        return ExecutionResult(
            executor_id=result.executor_id,
            sent=result.sent,
            intent=result.intent,
            message=result.message,
            metadata={
                "stage": "executor",
                "selected_executor": selected,
                "accepted": bool(bounded.accepted),
                "clipped": bool(bounded.clipped),
                "policy_reason": str(bounded.reason),
                **device_timing_metadata,
                **(
                    {
                        "scheduler": {
                            **(scheduler_metadata or {}),
                            **(
                                {"execution": scheduler_execution_metadata}
                                if scheduler_execution_metadata is not None
                                else {}
                            ),
                        }
                    }
                    if scheduler_metadata is not None or scheduler_execution_metadata is not None
                    else {}
                ),
            },
        )

    def _execute_direct(
        self,
        bounded: ControlOutput,
        *,
        expected_config_epoch: int,
        expected_selected: str,
        expected_single_command: bool,
    ) -> ExecutionResult:
        with self._executor_lock:
            with self._scheduler_lock:
                if (
                    self._config_epoch != expected_config_epoch
                    or self.selected != expected_selected
                    or self.single_command_per_observation != expected_single_command
                    or not (self.direct_output or self.single_command_per_observation)
                ):
                    return _direct_executor_superseded_result(
                        selected=expected_selected,
                        output=bounded,
                    )
                executor = self.executors[expected_selected]
                if expected_single_command:
                    return self.mouse_command_executor.execute_locked(
                        executor=executor,
                        command=bounded,
                    )
                stage = "direct_output"
                delivery_mode = "direct"
                if bounded.action == "move" and int(bounded.dx) == 0 and int(bounded.dy) == 0:
                    return ExecutionResult(
                        executor_id=expected_selected,
                        sent=False,
                        intent=bounded,
                        message="zero control command ignored",
                        metadata={
                            "stage": stage,
                            "delivery_mode": delivery_mode,
                            "selected_executor": expected_selected,
                            "scheduler_enabled": False,
                            "action": "zero_output",
                        },
                    )
                device_send_start_ts_ns = time.monotonic_ns()
                result = executor.execute(bounded)
                device_send_end_ts_ns = time.monotonic_ns()
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "stage": stage,
                "delivery_mode": delivery_mode,
                "selected_executor": expected_selected,
                "scheduler_enabled": False,
                "device_send_start_ts_ns": device_send_start_ts_ns,
                "device_send_end_ts_ns": device_send_end_ts_ns,
                "device_send_clock_domain": "monotonic",
            }
        )
        return replace(result, metadata=metadata)

    def tick_pending(
        self,
        *,
        now_s: float | None = None,
        recoil: RecoilDecision | None = None,
        recoil_source: ControlIntent | None = None,
    ) -> ExecutionResult:
        with self._scheduler_lock:
            single_command = self.single_command_per_observation
            latest_replace = self.latest_replace
            scheduler = None if single_command else self.scheduler
            selected = self.selected
            config_epoch = self._config_epoch
            output_enabled = self.output_enabled
            decision = (
                scheduler.tick(now_s=now_s)
                if output_enabled and scheduler is not None
                else None
            )
            submission_epoch = self._submission_epoch
        if not output_enabled:
            return _output_disabled_result(
                selected=selected,
                output=_scheduler_status_output("CONTROL_OUTPUT_DISABLED"),
                idle=True,
            )
        if scheduler is None:
            if single_command:
                return ExecutionResult(
                    executor_id=selected,
                    sent=False,
                    intent=_scheduler_status_output("single_command_executor_idle"),
                    message="no pending control command ready",
                    metadata={
                        "stage": "mouse_command_executor",
                        "delivery_mode": "single_command_per_observation",
                        "selected_executor": selected,
                        "scheduler_enabled": False,
                        "action": "no_pending_state",
                    },
                )
            return ExecutionResult(
                executor_id=selected,
                sent=False,
                intent=_scheduler_status_output("scheduler_required"),
                message="command scheduler required",
                metadata={
                    "stage": "scheduler_required",
                    "selected_executor": selected,
                },
            )
        assert decision is not None
        scheduler_metadata = decision.metadata
        tracking_output = decision.output
        output = tracking_output
        recoil_generation = (
            int(recoil_source.trajectory_generation)
            if recoil_source is not None
            and recoil_source.trajectory_generation is not None
            else None
        )
        tracking_generation = (
            int(tracking_output.trajectory_generation)
            if tracking_output is not None
            and tracking_output.trajectory_generation is not None
            else None
        )
        recoil_generation_matches = bool(
            tracking_output is None
            or recoil_generation is None
            or tracking_generation is None
            or recoil_generation == tracking_generation
        )
        recoil_owns_tick = bool(
            recoil is not None
            and recoil_generation_matches
            and recoil.state
            in {
                RecoilState.STARTUP,
                RecoilState.ACTIVE,
                RecoilState.HOLD,
                RecoilState.BRAKE,
            }
        )
        if recoil_owns_tick or (
            recoil is not None
            and recoil.state is RecoilState.STALE
            and tracking_output is not None
            and recoil_generation_matches
        ):
            template = tracking_output
            if template is None and recoil_source is not None:
                template = self.policy.apply(recoil_source)
                template = replace(template, dx=0, dy=0)
            if template is not None:
                mixed = ControlMixer().mix(
                    float(tracking_output.dx) if tracking_output is not None else 0.0,
                    float(tracking_output.dy) if tracking_output is not None else 0.0,
                    recoil,
                )
                output = replace(
                    template,
                    dx=int(mixed.final_x),
                    dy=int(mixed.final_y),
                    source_generation=(
                        int(recoil_source.trajectory_generation)
                        if recoil_source is not None
                        and recoil_source.trajectory_generation is not None
                        else template.trajectory_generation
                    ),
                    trigger_required=True if mixed.recoil_active else template.trigger_required,
                    trigger_active=True if mixed.recoil_active else template.trigger_active,
                    left_trigger_required=(
                        True if mixed.recoil_active else template.left_trigger_required
                    ),
                )
                scheduler_metadata = {
                    **scheduler_metadata,
                    "recoil_mixed": True,
                    "tracking_dx": mixed.tracking_x,
                    "tracking_dy": mixed.tracking_y,
                    "recoil_dy": mixed.recoil_y,
                    "mixed_dx": mixed.final_x,
                    "mixed_dy": mixed.final_y,
                    "recoil_owns_y": mixed.recoil_active,
                }
        elif recoil is not None and not recoil_generation_matches:
            scheduler_metadata = {
                **scheduler_metadata,
                "recoil_mixed": False,
                "recoil_generation_mismatch": True,
                "recoil_source_generation": recoil_generation,
                "tracking_source_generation": tracking_generation,
            }
        if output is None or (
            output.action == "move" and int(output.dx) == 0 and int(output.dy) == 0
        ):
            return ExecutionResult(
                executor_id=selected,
                sent=False,
                intent=_scheduler_status_output(
                    str(scheduler_metadata.get("action") or "scheduler_idle")
                ),
                message="no pending control command ready",
                metadata={
                    "stage": "scheduler",
                    "selected_executor": selected,
                    **scheduler_metadata,
                },
            )
        with self._executor_lock:
            with self._scheduler_lock:
                if (
                    self._config_epoch != config_epoch
                    or self.selected != selected
                    or self.single_command_per_observation
                    or self.latest_replace != latest_replace
                    or self.scheduler is not scheduler
                    or (latest_replace and self._submission_epoch != submission_epoch)
                ):
                    return _scheduler_superseded_result(
                        selected=selected,
                        output=output,
                    )
                self._actuation_sequence += 1
                output = replace(
                    output,
                    actuation_sequence=self._actuation_sequence,
                    source_generation=(
                        output.source_generation
                        if output.source_generation is not None
                        else output.trajectory_generation
                    ),
                )
                device_send_start_ts_ns = time.monotonic_ns()
                executor = self.executors[selected]
                result = (
                    self.mouse_command_executor.execute_locked(
                        executor=executor,
                        command=output,
                    )
                    if latest_replace
                    else executor.execute(output)
                )
                device_send_end_ts_ns = time.monotonic_ns()
                result_action = str((result.metadata or {}).get("action") or "")
                if latest_replace and result_action in {"blocked", "zero_output"}:
                    scheduler_execution_metadata = {
                        "stage": "scheduler",
                        "command_status": result_action,
                        "cooldown": False,
                    }
                else:
                    scheduler_execution_metadata = scheduler.record_execution_result(
                        sent=bool(result.sent),
                        message=str(result.message),
                        now_s=now_s,
                    )
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "delivery_mode": "latest_replace" if latest_replace else "scheduler",
                "scheduler_enabled": True,
                "device_send_start_ts_ns": device_send_start_ts_ns,
                "device_send_end_ts_ns": device_send_end_ts_ns,
                "device_send_clock_domain": "monotonic",
                "scheduler_send_delay_us": _scheduler_send_delay_us(
                    scheduler_metadata,
                    device_send_start_ts_ns,
                ),
            }
        )
        metadata["scheduler"] = {
            **scheduler_metadata,
            "execution": scheduler_execution_metadata,
        }
        return replace(result, metadata=metadata)

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
            "output_enabled": self.output_enabled,
            "direct_output": self.direct_output,
            "single_command_per_observation": self.single_command_per_observation,
            "latest_replace": self.latest_replace,
            "delivery_mode": (
                "latest_replace"
                if self.latest_replace
                else
                "single_command_per_observation"
                if self.single_command_per_observation
                else "direct"
                if self.direct_output
                else "legacy_scheduler"
            ),
            "scheduler": self._scheduler_status(),
            "executors": {
                executor_id: executor_status(executor)
                for executor_id, executor in self.executors.items()
            },
        }

    def read_buttons(self) -> dict[str, Any]:
        kmnet = self.executors.get("kmnet")
        reader = getattr(kmnet, "read_buttons", None)
        if not callable(reader):
            return {
                "available": False,
                "left": False,
                "right": False,
                "reason": "kmNet executor is unavailable",
            }
        return reader()

    def clear_scheduler(self, reason: str) -> None:
        with self._scheduler_lock:
            if not self.single_command_per_observation and self.scheduler is not None:
                self._submission_epoch += 1
                self.scheduler.clear(reason)

    def reset_mouse_command_executor(self) -> None:
        with self._scheduler_lock:
            self._actuation_sequence = 0
        self.mouse_command_executor.reset()

    def _scheduler_status(self) -> dict[str, Any]:
        with self._scheduler_lock:
            if not self.single_command_per_observation and self.scheduler is not None:
                return self.scheduler.status()
            return {
                "enabled": False,
                "direct_output": self.direct_output,
                "single_command_per_observation": self.single_command_per_observation,
                "latest_replace": self.latest_replace,
                "delivery_mode": (
                    "latest_replace"
                    if self.latest_replace
                    else
                    "single_command_per_observation"
                    if self.single_command_per_observation
                    else "direct"
                ),
            }


def policy_from_config(config: RuntimeConfig) -> ControlOutputPolicy:
    if config.control.active_algorithm == DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2:
        precise = config.control.dual_phase_atan_robust_predictive_v2
        maximum = int(
            math.ceil(
                max(
                    precise.atan.far.max_counts_per_update,
                    precise.atan.near.max_counts_per_update,
                )
            )
        )
        return ControlOutputPolicy(
            max_abs_dx=maximum,
            max_abs_dy=maximum,
            min_confidence=0.0,
        )
    step_x, step_y, interval_ms = _scheduler_delivery_config(config)
    capacity = plan_step_capacity(interval_ms)
    return ControlOutputPolicy(
        max_abs_dx=step_x * capacity,
        max_abs_dy=step_y * capacity,
        min_confidence=0.0,
    )


def scheduler_from_config(config: RuntimeConfig) -> CommandScheduler | None:
    latest_replace = bool(
        algorithm_definition(str(config.control.active_algorithm)).capabilities.scheduler_policy
        is SchedulerPolicy.LATEST_REPLACE
        or config.control.recoil.enabled
    )
    if not latest_replace and not bool(config.control.scheduler_enabled):
        return None
    if latest_replace and config.control.active_algorithm == DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2:
        precise = config.control.dual_phase_atan_robust_predictive_v2
        maximum = int(
            math.ceil(
                max(
                    precise.atan.far.max_counts_per_update,
                    precise.atan.near.max_counts_per_update,
                )
            )
        )
        step_x, step_y = maximum, maximum
        configured_interval_ms = float(config.control.scheduler_interval_ms)
    else:
        step_x, step_y, configured_interval_ms = _scheduler_delivery_config(config)
    interval_ms = max(1.0, min(10.0, configured_interval_ms))
    interval_s = interval_ms / 1000.0
    capacity = plan_step_capacity(interval_ms)
    expiry_s = (MAX_PLAN_DURATION_MS + interval_ms) / 1000.0
    return CommandScheduler(
        min_interval_s=interval_s,
        ttl_s=expiry_s,
        predicted_ttl_s=expiry_s,
        cancel_on_new_frame=True,
        cancel_on_direction_change=True,
        cancel_on_track_change=True,
        max_step_x=step_x,
        max_step_y=step_y,
        queue_hard_limit=capacity,
        device_error_cooldown_s=0.050,
    )


def _scheduler_delivery_config(config: RuntimeConfig) -> tuple[int, int, float]:
    return (
        int(config.control.scheduler_step_counts_x),
        int(config.control.scheduler_step_counts_y),
        float(config.control.scheduler_interval_ms),
    )


def _scheduler_status_output(reason: str) -> ControlOutput:
    return ControlOutput(
        dx=0,
        dy=0,
        action=None,
        confidence=0.0,
        source_id="scheduler",
        accepted=False,
        clipped=False,
        reason=reason,
    )


def _output_disabled_result(
    *,
    selected: str,
    output: ControlOutput,
    idle: bool = False,
) -> ExecutionResult:
    """Describe the global output gate without touching the live device."""
    return ExecutionResult(
        executor_id=selected,
        sent=False,
        intent=output,
        message=(
            "no pending control command ready"
            if idle
            else "mouse offset output disabled by runtime configuration"
        ),
        metadata={
            "stage": "output_gate",
            "selected_executor": selected,
            "action": "output_disabled",
            "block_reason": "CONTROL_OUTPUT_DISABLED",
            "output_enabled": False,
        },
    )


def _scheduler_superseded_result(
    *,
    selected: str,
    output: ControlOutput,
) -> ExecutionResult:
    return ExecutionResult(
        executor_id=selected,
        sent=False,
        intent=output,
        message="scheduled command discarded after executor reconfiguration",
        metadata={
            "stage": "scheduler",
            "selected_executor": selected,
            "action": "scheduler_superseded",
            "cancel_reason": "EXECUTOR_RECONFIGURED",
        },
    )


def _direct_executor_superseded_result(
    *,
    selected: str,
    output: ControlOutput,
) -> ExecutionResult:
    return ExecutionResult(
        executor_id=selected,
        sent=False,
        intent=output,
        message="direct command discarded after executor reconfiguration",
        metadata={
            "stage": "mouse_command_executor",
            "selected_executor": selected,
            "action": "executor_superseded",
            "cancel_reason": "EXECUTOR_RECONFIGURED",
        },
    )


def _scheduler_send_delay_us(metadata: dict[str, Any], send_start_ts_ns: int) -> float | None:
    scheduled_ts_ns = metadata.get("scheduled_ts_ns")
    if not isinstance(scheduled_ts_ns, (int, float)) or isinstance(scheduled_ts_ns, bool):
        return None
    return max(0.0, (int(send_start_ts_ns) - int(scheduled_ts_ns)) / 1000.0)
