from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    **{
        f"control.aim.role_y_ratios.{role}": _spec(
            f"control.aim.role_y_ratios.{role}",
            f"{label}瞄点纵向比例",
            0.22,
            0.0,
            1.0,
            0.01,
            "ratio",
            f"{label}类别检测框顶部向下的瞄点比例；X 固定为 bbox 中心。",
        )
        for role, label in (("head", "头部"), ("body", "身体"), ("other", "其他"))
    },
    "control.configured_actuation_delay_s": _spec(
        "control.configured_actuation_delay_s",
        "估计执行延迟",
        0.004,
        0.0,
        0.1,
        0.001,
        "s",
        "从控制计算到输入预计产生画面效果的配置估计。",
    ),
    "control.calibrated_angular.kp_x": _spec(
        "control.calibrated_angular.kp_x",
        "精确角度控制 Kp X",
        1.0,
        0.0,
        2.0,
        0.01,
        "",
        "预测角度误差的 X 轴比例增益。",
    ),
    "control.calibrated_angular.kp_y": _spec(
        "control.calibrated_angular.kp_y",
        "精确角度控制 Kp Y",
        1.0,
        0.0,
        2.0,
        0.01,
        "",
        "预测角度误差的 Y 轴比例增益。",
    ),
    "control.calibrated_angular.kd_x": _spec(
        "control.calibrated_angular.kd_x",
        "精确角度控制 Kd X",
        0.0,
        0.0,
        1.0,
        0.01,
        "s",
        "观测角度误差导数的 X 轴阻尼增益。",
    ),
    "control.calibrated_angular.kd_y": _spec(
        "control.calibrated_angular.kd_y",
        "精确角度控制 Kd Y",
        0.0,
        0.0,
        1.0,
        0.01,
        "s",
        "观测角度误差导数的 Y 轴阻尼增益。",
    ),
    "control.calibrated_angular.d_ema_alpha": _spec(
        "control.calibrated_angular.d_ema_alpha",
        "精确角度控制 D 项 EMA",
        0.30,
        0.01,
        1.0,
        0.01,
        "ratio",
        "D 项新观测权重；越小越平滑但延迟越大。",
    ),
    "control.calibrated_angular.max_angle_step_x_deg": _spec(
        "control.calibrated_angular.max_angle_step_x_deg",
        "精确角度控制 X 最大角度步长",
        2.0,
        0.0001,
        180.0,
        0.0001,
        "deg",
        "每个新观测允许生成的 X 最大修正角度。",
    ),
    "control.calibrated_angular.max_angle_step_y_deg": _spec(
        "control.calibrated_angular.max_angle_step_y_deg",
        "精确角度控制 Y 最大角度步长",
        1.5,
        0.0001,
        180.0,
        0.0001,
        "deg",
        "每个新观测允许生成的 Y 最大修正角度。",
    ),
    "control.universal_saturated.response_scale_x_px": _spec(
        "control.universal_saturated.response_scale_x_px",
        "通用水平响应尺度",
        80.0,
        0.1,
        4000.0,
        0.1,
        "px",
        "X 误差进入明显非线性压缩的像素尺度；数值越小，近中心响应越强。",
    ),
    "control.universal_saturated.response_scale_y_px": _spec(
        "control.universal_saturated.response_scale_y_px",
        "通用垂直响应尺度",
        60.0,
        0.1,
        4000.0,
        0.1,
        "px",
        "Y 误差进入明显非线性压缩的像素尺度；数值越小，近中心响应越强。",
    ),
    "control.universal_saturated.max_step_x_counts": _spec(
        "control.universal_saturated.max_step_x_counts",
        "通用最大水平移动",
        50.0,
        0.1,
        1000.0,
        0.1,
        "counts",
        "通用饱和模式 X 输出渐近上限。",
    ),
    "control.universal_saturated.max_step_y_counts": _spec(
        "control.universal_saturated.max_step_y_counts",
        "通用最大垂直移动",
        40.0,
        0.1,
        1000.0,
        0.1,
        "counts",
        "通用饱和模式 Y 输出渐近上限。",
    ),
    "control.shared.deadzone_x_px": _spec(
        "control.shared.deadzone_x_px",
        "共享 X 到位阈值",
        4.0,
        0.0,
        10.0,
        0.1,
        "px",
        "观测瞄点进入 X 到位区后停止该轴；退出阈值由系统自动增加滞回。",
    ),
    "control.shared.deadzone_y_px": _spec(
        "control.shared.deadzone_y_px",
        "共享 Y 到位阈值",
        4.0,
        0.0,
        10.0,
        0.1,
        "px",
        "观测瞄点进入 Y 到位区后停止该轴；退出阈值由系统自动增加滞回。",
    ),
    "control.shared.max_count_slew_x": _spec(
        "control.shared.max_count_slew_x",
        "共享 X counts 增长限制",
        10.0,
        0.1,
        1000.0,
        0.1,
        "counts",
        "相邻观测 X 输出允许增加的最大 counts；减速立即生效，反向前先归零。",
    ),
    "control.shared.max_count_slew_y": _spec(
        "control.shared.max_count_slew_y",
        "共享 Y counts 增长限制",
        8.0,
        0.1,
        1000.0,
        0.1,
        "counts",
        "相邻观测 Y 输出允许增加的最大 counts；减速立即生效，反向前先归零。",
    ),
    "control.shared.trigger_activation_delay_ms": _spec(
        "control.shared.trigger_activation_delay_ms",
        "触发后启动延迟",
        0.0,
        0.0,
        1000.0,
        1.0,
        "ms",
        "真实鼠标触发持续达到该时间后才允许瞄准算法输出。",
    ),
    "control.shared.recoil_start_delay_ms": _spec(
        "control.shared.recoil_start_delay_ms",
        "压枪启动延迟",
        0.0,
        0.0,
        1000.0,
        1.0,
        "ms",
        "左键按下后等待多久开始 Y 轴压枪前馈。",
    ),
    "control.shared.recoil_y_counts_per_observation": _spec(
        "control.shared.recoil_y_counts_per_observation",
        "单观测固定 Y 压枪",
        0.0,
        0.0,
        20.0,
        0.1,
        "counts",
        "启动延迟后，每个新鲜目标观测固定加入的反向 Y counts；小数由独立余量累计。",
    ),
    "control.scheduler_step_counts_x": _spec(
        "control.scheduler_step_counts_x",
        "Scheduler X 单步",
        8.0,
        1.0,
        20.0,
        1.0,
        "counts",
        "Scheduler 每步允许发送的 X 轴最大 counts。",
    ),
    "control.scheduler_step_counts_y": _spec(
        "control.scheduler_step_counts_y",
        "Scheduler Y 单步",
        8.0,
        1.0,
        20.0,
        1.0,
        "counts",
        "Scheduler 每步允许发送的 Y 轴最大 counts。",
    ),
    "control.scheduler_interval_ms": _spec(
        "control.scheduler_interval_ms",
        "Scheduler 间隔",
        4.0,
        1.0,
        10.0,
        0.1,
        "ms",
        "Scheduler 小步发送间隔。",
    ),
    "control.target_fov_radius_px": _spec(
        "control.target_fov_radius_px",
        "目标选择半径",
        180.0,
        1.0,
        2000.0,
        1.0,
        "px",
        "以 640 ROI 为基准的目标候选半径，运行时按实际 ROI 尺寸同比缩放。",
    ),
    "control.candidate_selection_class_weight": _spec(
        "control.candidate_selection_class_weight",
        "目标类别权重",
        0.55,
        0.0,
        2.0,
        0.01,
        "ratio",
        "综合目标分数中类别偏好的相对权重。",
    ),
    "control.candidate_selection_quality_weight": _spec(
        "control.candidate_selection_quality_weight",
        "目标质量权重",
        0.05,
        0.0,
        2.0,
        0.01,
        "ratio",
        "综合目标分数中检测与 Track 质量的相对权重。",
    ),
    "control.candidate_selection_distance_weight": _spec(
        "control.candidate_selection_distance_weight",
        "目标距离权重",
        0.40,
        0.0,
        2.0,
        0.01,
        "ratio",
        "综合目标分数中归一化中心距离的相对权重。",
    ),
    "control.target_switch_delay_ms": _spec(
        "control.target_switch_delay_ms",
        "目标切换延迟",
        50.0,
        0.0,
        500.0,
        1.0,
        "ms",
        "新目标持续满足切换条件后才提交切换的时间。",
    ),
    "control.calibrated_angular.fov_x_deg": _spec(
        "control.calibrated_angular.fov_x_deg",
        "水平 FOVX",
        105.0,
        30.0,
        179.0,
        0.1,
        "deg",
        "完整控制投影空间的水平视场角。",
    ),
    "control.calibrated_angular.counts_per_360_x": _spec(
        "control.calibrated_angular.counts_per_360_x",
        "水平每圈 counts",
        9980.0,
        1.0,
        100000.0,
        1.0,
        "counts",
        "X 轴旋转一整圈对应的设备 counts。",
    ),
    "control.calibrated_angular.counts_per_360_y": _spec(
        "control.calibrated_angular.counts_per_360_y",
        "垂直每圈 counts",
        9980.0,
        1.0,
        100000.0,
        1.0,
        "counts",
        "Y 轴旋转一整圈对应的设备 counts。",
    ),
}


for _legacy_prefix, _namespaced_prefix in (
    ("control.calibrated_angular.", "control.algorithms.calibrated_angular."),
    ("control.universal_saturated.", "control.algorithms.universal_saturated."),
):
    for _key, _value in tuple(CONTROL_PARAM_SPECS.items()):
        if _key.startswith(_legacy_prefix):
            _namespaced_key = _namespaced_prefix + _key.removeprefix(_legacy_prefix)
            CONTROL_PARAM_SPECS[_namespaced_key] = replace(_value, key=_namespaced_key)


CONTROL_PARAM_SPECS.update(
    {
        "control.algorithms.dual_phase_atan_robust_predictive_v2.projection.fov_x_deg": _spec(
            "control.algorithms.dual_phase_atan_robust_predictive_v2.projection.fov_x_deg",
            "稳健预测控制水平 FOVX",
            105.0,
            30.0,
            179.0,
            0.1,
            "deg",
            "用于将完整画面像素误差转换为角度误差。",
        ),
        "control.algorithms.dual_phase_atan_robust_predictive_v2.projection.counts_per_360": _spec(
            "control.algorithms.dual_phase_atan_robust_predictive_v2.projection.counts_per_360",
            "稳健预测控制每圈 counts",
            9980.0,
            1.0,
            100000.0,
            1.0,
            "counts",
            "设备旋转一整圈所需的标定 counts。",
        ),
        "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.lead_frames": _spec(
            "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.lead_frames",
            "稳健预测控制前瞻帧数",
            1.0,
            0.0,
            10.0,
            0.01,
            "frames",
            "与平均捕获 dt 和平滑目标速度相乘；0 完全关闭位置预测。",
        ),
        "control.algorithms.dual_phase_atan_robust_predictive_v2.mode.near_threshold_px": _spec(
            "control.algorithms.dual_phase_atan_robust_predictive_v2.mode.near_threshold_px",
            "稳健预测控制 NEAR 阈值",
            12.0,
            0.0,
            1000.0,
            0.1,
            "px",
            "测量误差距离不大于该值时使用 NEAR，否则使用 FAR。",
        ),
        "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.scale_counts": _spec(
            "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.scale_counts",
            "稳健预测控制 Atan 共享尺度",
            1024.0,
            0.1,
            10000.0,
            0.1,
            "counts",
            "FAR 与 NEAR 共同使用的 counts 域非线性压缩尺度。",
        ),
        "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.far.kp": _spec(
            "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.far.kp",
            "稳健预测控制远距离跟进强度",
            0.90,
            0.001,
            0.999,
            0.001,
            "ratio",
            "FAR 阶段完整修正 counts 的 Atan 比例增益。",
        ),
        "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.near.kp": _spec(
            "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.near.kp",
            "稳健预测控制近距离跟随强度",
            0.30,
            0.001,
            0.999,
            0.001,
            "ratio",
            "NEAR 阶段无死区小幅持续修正增益。",
        ),
        "control.algorithms.dual_phase_atan_robust_predictive_v2.velocity.smoothing_frames": _spec(
            "control.algorithms.dual_phase_atan_robust_predictive_v2.velocity.smoothing_frames",
            "稳健预测控制速度平滑帧数",
            3.0,
            0.1,
            20.0,
            0.1,
            "frames",
            "速度 EMA 的等效帧窗口；内部仍按真实捕获 dt 处理丢帧和抖动。",
        ),
    }
)


def param_schema_for(path: str) -> dict[str, Any]:
    spec = CONTROL_PARAM_SPECS.get(path)
    return spec.to_schema() if spec is not None else {}
