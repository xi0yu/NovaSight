from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


UNIVERSAL_SATURATED = "universal_saturated"
CALIBRATED_ANGULAR = "calibrated_angular"
DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2 = "dual_phase_atan_robust_predictive_v2"
DEFAULT_ACTIVE_ALGORITHM_ID = DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2


class SchedulerPolicy(str, Enum):
    OPTIONAL = "optional"
    CONFIGURABLE = "configurable"
    LATEST_REPLACE = "latest_replace"


@dataclass(frozen=True, slots=True)
class AlgorithmCapabilities:
    requires_calibration: bool
    supports_scheduler: bool
    supports_prediction: bool
    scheduler_policy: SchedulerPolicy
    scheduler_policy_ready: bool


@dataclass(frozen=True, slots=True)
class AlgorithmDefinition:
    algorithm_id: str
    product_name: str
    product_description: str
    capabilities: AlgorithmCapabilities


_ALGORITHM_DEFINITIONS = (
    AlgorithmDefinition(
        algorithm_id=UNIVERSAL_SATURATED,
        product_name="通用控制",
        product_description="无需精确游戏参数的快速适配控制。",
        capabilities=AlgorithmCapabilities(
            requires_calibration=False,
            supports_scheduler=True,
            supports_prediction=False,
            scheduler_policy=SchedulerPolicy.OPTIONAL,
            scheduler_policy_ready=True,
        ),
    ),
    AlgorithmDefinition(
        algorithm_id=CALIBRATED_ANGULAR,
        product_name="精确角度控制",
        product_description="依赖 FOV 和 counts_per_360 标定的角度控制。",
        capabilities=AlgorithmCapabilities(
            requires_calibration=True,
            supports_scheduler=True,
            supports_prediction=False,
            scheduler_policy=SchedulerPolicy.CONFIGURABLE,
            scheduler_policy_ready=True,
        ),
    ),
    AlgorithmDefinition(
        algorithm_id=DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2,
        product_name="双阶段 Atan 控制",
        product_description="仅使用当前观测误差、角度投影和 Atan 响应曲线。",
        capabilities=AlgorithmCapabilities(
            requires_calibration=True,
            supports_scheduler=True,
            supports_prediction=False,
            scheduler_policy=SchedulerPolicy.LATEST_REPLACE,
            scheduler_policy_ready=True,
        ),
    ),
)
_ALGORITHMS_BY_ID = {item.algorithm_id: item for item in _ALGORITHM_DEFINITIONS}


def supported_algorithm_ids() -> tuple[str, ...]:
    return tuple(item.algorithm_id for item in _ALGORITHM_DEFINITIONS)


def algorithm_definition(algorithm_id: str) -> AlgorithmDefinition:
    try:
        return _ALGORITHMS_BY_ID[str(algorithm_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported control algorithm: {algorithm_id}") from exc


class AlgorithmRegistry:
    """Owns the one controller that is allowed to consume observations."""

    def __init__(self, algorithm_id: str, controller: Any) -> None:
        self._definition = algorithm_definition(algorithm_id)
        self._controller = controller

    @property
    def active_algorithm_id(self) -> str:
        return self._definition.algorithm_id

    @property
    def active_definition(self) -> AlgorithmDefinition:
        return self._definition

    @property
    def active_controller(self) -> Any:
        return self._controller

    def is_active(self, algorithm_id: str) -> bool:
        return self.active_algorithm_id == str(algorithm_id)

    def reset(self) -> None:
        reset = getattr(self._controller, "reset", None)
        if callable(reset):
            reset()
