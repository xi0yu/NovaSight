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
    "control.calibrated_angular.kp_x": _spec("control.calibrated_angular.kp_x", "精确标定 Kp X", 1.0, 0.0, 2.0, 0.01, "", "预测角度误差的 X 轴比例增益。"),
    "control.calibrated_angular.kp_y": _spec("control.calibrated_angular.kp_y", "精确标定 Kp Y", 1.0, 0.0, 2.0, 0.01, "", "预测角度误差的 Y 轴比例增益。"),
    "control.calibrated_angular.kd_x": _spec("control.calibrated_angular.kd_x", "精确标定 Kd X", 0.0, 0.0, 1.0, 0.01, "s", "观测角度误差导数的 X 轴阻尼增益。"),
    "control.calibrated_angular.kd_y": _spec("control.calibrated_angular.kd_y", "精确标定 Kd Y", 0.0, 0.0, 1.0, 0.01, "s", "观测角度误差导数的 Y 轴阻尼增益。"),
    "control.calibrated_angular.d_ema_alpha": _spec("control.calibrated_angular.d_ema_alpha", "精确标定 D 项 EMA", 0.30, 0.01, 1.0, 0.01, "ratio", "D 项新观测权重；越小越平滑但延迟越大。"),
    "control.calibrated_angular.max_angle_step_x_deg": _spec("control.calibrated_angular.max_angle_step_x_deg", "精确标定 X 最大角度步长", 2.0, 0.0001, 180.0, 0.0001, "deg", "每个新观测允许生成的 X 最大修正角度。"),
    "control.calibrated_angular.max_angle_step_y_deg": _spec("control.calibrated_angular.max_angle_step_y_deg", "精确标定 Y 最大角度步长", 1.5, 0.0001, 180.0, 0.0001, "deg", "每个新观测允许生成的 Y 最大修正角度。"),
    "control.universal_saturated.response_scale_x_px": _spec("control.universal_saturated.response_scale_x_px", "通用水平响应尺度", 80.0, 0.1, 4000.0, 0.1, "px", "X 误差进入明显非线性压缩的像素尺度；数值越小，近中心响应越强。"),
    "control.universal_saturated.response_scale_y_px": _spec("control.universal_saturated.response_scale_y_px", "通用垂直响应尺度", 60.0, 0.1, 4000.0, 0.1, "px", "Y 误差进入明显非线性压缩的像素尺度；数值越小，近中心响应越强。"),
    "control.universal_saturated.max_step_x_counts": _spec("control.universal_saturated.max_step_x_counts", "通用最大水平移动", 50.0, 0.1, 1000.0, 0.1, "counts", "通用饱和模式 X 输出渐近上限。"),
    "control.universal_saturated.max_step_y_counts": _spec("control.universal_saturated.max_step_y_counts", "通用最大垂直移动", 40.0, 0.1, 1000.0, 0.1, "counts", "通用饱和模式 Y 输出渐近上限。"),
    "control.ttbox_pid_atan.response_scale_x_px": _spec("control.ttbox_pid_atan.response_scale_x_px", "ttbox_pid_atan 水平响应尺度", 256.0, 0.1, 4000.0, 0.1, "px", "atan 分母标度；像素误差除以它再进入 atan，控制近中心斜率。"),
    "control.ttbox_pid_atan.response_scale_y_px": _spec("control.ttbox_pid_atan.response_scale_y_px", "ttbox_pid_atan 垂直响应尺度", 256.0, 0.1, 4000.0, 0.1, "px", "Y 轴 atan 分母标度。"),
    "control.ttbox_pid_atan.per_frame_gain_x": _spec("control.ttbox_pid_atan.per_frame_gain_x", "ttbox_pid_atan X 每帧增益", 0.1, 0.001, 2.0, 0.001, "ratio", "X 轴每帧衰减；dt 归一化后会被 dt/nominal 缩放。"),
    "control.ttbox_pid_atan.per_frame_gain_y": _spec("control.ttbox_pid_atan.per_frame_gain_y", "ttbox_pid_atan Y 每帧增益", 0.1, 0.001, 2.0, 0.001, "ratio", "Y 轴每帧衰减。"),
    "control.ttbox_pid_atan.max_step_x_counts": _spec("control.ttbox_pid_atan.max_step_x_counts", "ttbox_pid_atan X 最大 counts", 50.0, 0.1, 1000.0, 0.1, "counts", "单帧 X 输出的硬钳上限。"),
    "control.ttbox_pid_atan.max_step_y_counts": _spec("control.ttbox_pid_atan.max_step_y_counts", "ttbox_pid_atan Y 最大 counts", 40.0, 0.1, 1000.0, 0.1, "counts", "单帧 Y 输出的硬钳上限。"),
    "control.ttbox_pid_atan.normalize_to_dt": _spec("control.ttbox_pid_atan.normalize_to_dt", "ttbox_pid_atan 按 dt 归一", 1.0, 0.0, 1.0, 1.0, "bool", "是否按 dt/nominal_dt_s 缩放每帧增益，使刷新率变化时响应一致。"),
    "control.ttbox_pid_atan.nominal_dt_s": _spec("control.ttbox_pid_atan.nominal_dt_s", "ttbox_pid_atan 标称帧间隔", 0.016, 0.001, 0.2, 0.001, "s", "用于 dt 归一的基准帧间隔；60Hz 取 0.0166，120Hz 取 0.0083。"),
    "control.shared.deadzone_x_px": _spec("control.shared.deadzone_x_px", "共享 X 到位阈值", 4.0, 0.0, 10.0, 0.1, "px", "观测瞄点进入 X 到位区后停止该轴；退出阈值由系统自动增加滞回。"),
    "control.shared.deadzone_y_px": _spec("control.shared.deadzone_y_px", "共享 Y 到位阈值", 4.0, 0.0, 10.0, 0.1, "px", "观测瞄点进入 Y 到位区后停止该轴；退出阈值由系统自动增加滞回。"),
    "control.shared.max_count_slew_x": _spec("control.shared.max_count_slew_x", "共享 X counts 变化限制", 10.0, 0.1, 1000.0, 0.1, "counts", "相邻观测 X 输出的最大 counts 变化量。"),
    "control.shared.max_count_slew_y": _spec("control.shared.max_count_slew_y", "共享 Y counts 变化限制", 8.0, 0.1, 1000.0, 0.1, "counts", "相邻观测 Y 输出的最大 counts 变化量。"),
    "control.shared.trigger_activation_delay_ms": _spec("control.shared.trigger_activation_delay_ms", "触发后启动延迟", 0.0, 0.0, 1000.0, 1.0, "ms", "真实鼠标触发持续达到该时间后才允许瞄准算法输出。"),
    "control.shared.recoil_start_delay_ms": _spec("control.shared.recoil_start_delay_ms", "压枪启动延迟", 0.0, 0.0, 1000.0, 1.0, "ms", "左键按下后等待多久开始 Y 轴压枪前馈。"),
    "control.shared.recoil_y_rate_counts_s": _spec("control.shared.recoil_y_rate_counts_s", "Y 压枪速率", 0.0, 0.0, 5000.0, 1.0, "counts/s", "正值表示持续向设备 Y 正方向补偿。"),
    "control.shared.recoil_ramp_up_ms": _spec("control.shared.recoil_ramp_up_ms", "压枪渐入", 120.0, 0.0, 2000.0, 1.0, "ms", "压枪从零平滑增加到配置速率所需时间。"),
    "control.shared.recoil_max_counts_per_observation": _spec("control.shared.recoil_max_counts_per_observation", "单观测最大压枪", 8.0, 0.1, 20.0, 0.1, "counts", "限制一次新视觉观测可加入的压枪预算。"),
    "control.scheduler_step_counts_x": _spec("control.scheduler_step_counts_x", "Scheduler X 单步", 8.0, 1.0, 20.0, 1.0, "counts", "Scheduler 每步允许发送的 X 轴最大 counts。"),
    "control.scheduler_step_counts_y": _spec("control.scheduler_step_counts_y", "Scheduler Y 单步", 8.0, 1.0, 20.0, 1.0, "counts", "Scheduler 每步允许发送的 Y 轴最大 counts。"),
    "control.scheduler_interval_ms": _spec("control.scheduler_interval_ms", "Scheduler 间隔", 4.0, 1.0, 10.0, 0.1, "ms", "Scheduler 小步发送间隔。"),
    "control.target_fov_radius_px": _spec("control.target_fov_radius_px", "目标选择半径", 180.0, 1.0, 2000.0, 1.0, "px", "以 ROI 中心为圆心的目标候选范围。"),
    "control.min_confidence": _spec("control.min_confidence", "最低置信度", 0.25, 0.10, 0.99, 0.01, "ratio", "允许进入 Tracker 和控制链的最低检测置信度。"),
    "control.target_switch_delay_ms": _spec("control.target_switch_delay_ms", "目标切换延迟", 50.0, 0.0, 500.0, 1.0, "ms", "新目标持续满足切换条件后才提交切换的时间。"),
    "control.calibrated_angular.fov_x_deg": _spec("control.calibrated_angular.fov_x_deg", "水平 FOVX", 105.0, 30.0, 179.0, 0.1, "deg", "完整控制投影空间的水平视场角。"),
    "control.calibrated_angular.counts_per_360_x": _spec("control.calibrated_angular.counts_per_360_x", "水平每圈 counts", 9980.0, 1.0, 100000.0, 1.0, "counts", "X 轴旋转一整圈对应的设备 counts。"),
    "control.calibrated_angular.counts_per_360_y": _spec("control.calibrated_angular.counts_per_360_y", "垂直每圈 counts", 9980.0, 1.0, 100000.0, 1.0, "counts", "Y 轴旋转一整圈对应的设备 counts。"),
}


def param_schema_for(path: str) -> dict[str, Any]:
    spec = CONTROL_PARAM_SPECS.get(path)
    return spec.to_schema() if spec is not None else {}
