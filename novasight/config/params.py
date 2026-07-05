from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParamSpec:
    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    recommended_min: float
    recommended_max: float
    step: float
    unit: str
    description: str

    def to_schema(self) -> dict[str, Any]:
        return asdict(self)


CONTROL_PARAM_SPECS: dict[str, ParamSpec] = {
    "control.pid_kp_x": ParamSpec(
        key="control.pid_kp_x",
        label="Kp X",
        default=0.35,
        minimum=0.0,
        maximum=2.0,
        recommended_min=0.20,
        recommended_max=0.60,
        step=0.01,
        unit="",
        description="X 轴比例增益，越大横向响应越快，但更容易过冲。",
    ),
    "control.pid_kp_y": ParamSpec(
        key="control.pid_kp_y",
        label="Kp Y",
        default=0.24,
        minimum=0.0,
        maximum=2.0,
        recommended_min=0.15,
        recommended_max=0.45,
        step=0.01,
        unit="",
        description="Y 轴比例增益，通常应低于 X 轴，避免垂直方向抖动。",
    ),
    "control.pid_kd": ParamSpec(
        key="control.pid_kd",
        label="Kd",
        default=0.10,
        minimum=0.0,
        maximum=1.0,
        recommended_min=0.05,
        recommended_max=0.25,
        step=0.01,
        unit="",
        description="阻尼系数，用于抑制接近目标时的过冲。",
    ),
    "control.deadzone_counts": ParamSpec(
        key="control.deadzone_counts",
        label="Dead Zone",
        default=1.0,
        minimum=0.0,
        maximum=30.0,
        recommended_min=0.0,
        recommended_max=4.0,
        step=1.0,
        unit="count",
        description="误差进入该范围后停止输出，避免在目标附近抖动。",
    ),
    "control.kp_x_move_max": ParamSpec(
        key="control.kp_x_move_max",
        label="Max Step X",
        default=150.0,
        minimum=1.0,
        maximum=500.0,
        recommended_min=40.0,
        recommended_max=180.0,
        step=1.0,
        unit="count",
        description="X 轴单次控制输出上限。",
    ),
    "control.kp_y_move_max": ParamSpec(
        key="control.kp_y_move_max",
        label="Max Step Y",
        default=30.0,
        minimum=1.0,
        maximum=200.0,
        recommended_min=10.0,
        recommended_max=60.0,
        step=1.0,
        unit="count",
        description="Y 轴单次控制输出上限。",
    ),
    "control.straight_in_deadzone": ParamSpec(
        key="control.straight_in_deadzone",
        label="Straight Dead Zone",
        default=8.0,
        minimum=0.0,
        maximum=100.0,
        recommended_min=2.0,
        recommended_max=12.0,
        step=1.0,
        unit="px",
        description="推荐算法的输入死区，目标误差小于该像素值时停止该轴输出。",
    ),
    "control.straight_max_step": ParamSpec(
        key="control.straight_max_step",
        label="Straight Max Step",
        default=80.0,
        minimum=1.0,
        maximum=500.0,
        recommended_min=40.0,
        recommended_max=140.0,
        step=1.0,
        unit="count",
        description="推荐算法精修阶段的单次输出上限。",
    ),
    "control.straight_fov_deg": ParamSpec(
        key="control.straight_fov_deg",
        label="FOV",
        default=105.0,
        minimum=1.0,
        maximum=179.0,
        recommended_min=90.0,
        recommended_max=115.0,
        step=1.0,
        unit="deg",
        description="像素误差转换为角度时使用的水平视场角。",
    ),
    "control.straight_c360": ParamSpec(
        key="control.straight_c360",
        label="c360",
        default=9980.0,
        minimum=100.0,
        maximum=50000.0,
        recommended_min=8000.0,
        recommended_max=12000.0,
        step=20.0,
        unit="count",
        description="转一圈对应的 counts，用于把角度转换为 kmNet 控制量。",
    ),
}


def param_schema_for(path: str) -> dict[str, Any]:
    spec = CONTROL_PARAM_SPECS.get(path)
    return spec.to_schema() if spec is not None else {}
