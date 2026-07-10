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


def _spec(
    key: str,
    label: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    unit: str,
    description: str,
) -> ParamSpec:
    return ParamSpec(
        key=key,
        label=label,
        default=default,
        minimum=minimum,
        maximum=maximum,
        recommended_min=minimum,
        recommended_max=maximum,
        step=step,
        unit=unit,
        description=description,
    )


CONTROL_PARAM_SPECS: dict[str, ParamSpec] = {
    "control.aim.y_ratio": _spec("control.aim.y_ratio", "瞄点纵向比例", 0.22, 0.0, 1.0, 0.01, "ratio", "目标框顶部向下的瞄点比例；X 固定为 bbox 中心。"),
    "control.configured_actuation_delay_s": _spec("control.configured_actuation_delay_s", "估计执行延迟", 0.004, 0.0, 0.1, 0.001, "s", "从控制计算到输入预计产生画面效果的配置估计。"),
    "control.prediction_strength": _spec("control.prediction_strength", "预测强度", 1.0, 0.0, 1.5, 0.01, "ratio", "Kalman 速度外推位移的应用比例。"),
    "control.kp_x": _spec("control.kp_x", "Kp X", 0.35, 0.0, 2.0, 0.01, "", "预测角度误差的 X 轴比例增益。"),
    "control.kp_y": _spec("control.kp_y", "Kp Y", 0.24, 0.0, 2.0, 0.01, "", "预测角度误差的 Y 轴比例增益。"),
    "control.kd_x": _spec("control.kd_x", "Kd X", 0.0, 0.0, 1.0, 0.01, "s", "观测角度误差导数的 X 轴阻尼增益。"),
    "control.kd_y": _spec("control.kd_y", "Kd Y", 0.0, 0.0, 1.0, 0.01, "s", "观测角度误差导数的 Y 轴阻尼增益。"),
    "control.d_ema_alpha": _spec("control.d_ema_alpha", "D 项 EMA", 0.25, 0.01, 1.0, 0.01, "ratio", "D 项新观测权重；越小越平滑但延迟越大。"),
    "control.deadzone_px_x": _spec("control.deadzone_px_x", "X 像素死区", 0.0, 0.0, 10.0, 0.1, "px", "观测瞄点的 X 轴控制死区。"),
    "control.deadzone_px_y": _spec("control.deadzone_px_y", "Y 像素死区", 0.0, 0.0, 10.0, 0.1, "px", "观测瞄点的 Y 轴控制死区。"),
    "control.max_output_rad_x": _spec("control.max_output_rad_x", "X 单次最大角度", 0.0524, 0.0001, 1.0, 0.0001, "rad", "每个新观测生成的 X 轴最大修正角度。"),
    "control.max_output_rad_y": _spec("control.max_output_rad_y", "Y 单次最大角度", 0.0524, 0.0001, 1.0, 0.0001, "rad", "每个新观测生成的 Y 轴最大修正角度。"),
    "control.max_output_rate_rad_s_x": _spec("control.max_output_rate_rad_s_x", "X 输出变化率", 2.0, 0.0001, 100.0, 0.0001, "rad/s", "X 轴相邻观测输出角度的最大变化率。"),
    "control.max_output_rate_rad_s_y": _spec("control.max_output_rate_rad_s_y", "Y 输出变化率", 2.0, 0.0001, 100.0, 0.0001, "rad/s", "Y 轴相邻观测输出角度的最大变化率。"),
    "control.scheduler_step_counts_x": _spec("control.scheduler_step_counts_x", "Scheduler X 单步", 20.0, 1.0, 20.0, 1.0, "counts", "Scheduler 每步允许发送的 X 轴最大 counts。"),
    "control.scheduler_step_counts_y": _spec("control.scheduler_step_counts_y", "Scheduler Y 单步", 20.0, 1.0, 20.0, 1.0, "counts", "Scheduler 每步允许发送的 Y 轴最大 counts。"),
    "control.scheduler_interval_ms": _spec("control.scheduler_interval_ms", "Scheduler 间隔", 4.0, 1.0, 10.0, 0.1, "ms", "Scheduler 小步发送间隔。"),
    "control.target_fov_radius_px": _spec("control.target_fov_radius_px", "目标选择半径", 180.0, 1.0, 2000.0, 1.0, "px", "以 ROI 中心为圆心的目标候选范围。"),
    "control.min_confidence": _spec("control.min_confidence", "最低置信度", 0.25, 0.10, 0.99, 0.01, "ratio", "允许进入 Tracker 和控制链的最低检测置信度。"),
    "control.target_switch_delay_ms": _spec("control.target_switch_delay_ms", "目标切换延迟", 50.0, 0.0, 500.0, 1.0, "ms", "新目标持续满足切换条件后才提交切换的时间。"),
    "control.lost_target_timeout_ms": _spec("control.lost_target_timeout_ms", "目标丢失超时", 120.0, 0.0, 200.0, 1.0, "ms", "短时漏检时允许 Kalman 维持目标的最长时间。"),
    "calibration.fov_x_deg": _spec("calibration.fov_x_deg", "水平 FOVX", 105.0, 30.0, 179.0, 0.1, "deg", "完整控制投影空间的水平视场角。"),
    "calibration.counts_per_360_x": _spec("calibration.counts_per_360_x", "水平每圈 counts", 9980.0, 1.0, 100000.0, 1.0, "counts", "X 轴旋转一整圈对应的设备 counts。"),
    "calibration.counts_per_360_y": _spec("calibration.counts_per_360_y", "垂直每圈 counts", 9980.0, 1.0, 100000.0, 1.0, "counts", "Y 轴旋转一整圈对应的设备 counts。"),
}


def param_schema_for(path: str) -> dict[str, Any]:
    spec = CONTROL_PARAM_SPECS.get(path)
    return spec.to_schema() if spec is not None else {}
