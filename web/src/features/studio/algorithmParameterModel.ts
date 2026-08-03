import type { RuntimeConfigValue } from "../../api";
import type {
  ParameterApplyMode,
  ParameterNumberKind,
  ParameterRiskLevel
} from "./StudioControls";

export type StudioNumberParameter<Field extends string> = {
  key: Field;
  label: string;
  detail: string;
  value: number;
  min: number;
  max: number;
  recommendedMin?: number;
  recommendedMax?: number;
  step: number;
  unit?: string;
  kind?: ParameterNumberKind;
  applyMode?: ParameterApplyMode;
  riskLevel?: ParameterRiskLevel;
  transform?: (value: number) => RuntimeConfigValue;
};

export type AlgorithmNumberParameter = StudioNumberParameter<DualPhasePipelineField>;
export type TargetingNumberParameter = StudioNumberParameter<TargetingPipelineField>;

export type DualPhasePipelineField =
  | "freshness_threshold_ms"
  | "projection_fov_x_deg"
  | "projection_counts_per_360"
  | "near_threshold_px"
  | "far_kp"
  | "near_kp"
  | "atan_scale_counts"
  | "far_max_counts_per_update"
  | "near_max_counts_per_update"
  | "prediction_enabled"
  | "velocity_smoothing_frames"
  | "velocity_history_reset_gap_ms"
  | "velocity_spread_base_px_ms"
  | "velocity_spread_relative"
  | "velocity_change_base_px_ms"
  | "velocity_change_relative"
  | "prediction_lead_frames"
  | "prediction_far_absolute_cap_px"
  | "prediction_far_base_cap_px"
  | "prediction_far_relative_cap"
  | "prediction_near_absolute_cap_px"
  | "prediction_near_base_cap_px"
  | "prediction_near_relative_cap"
  | "arrival_radius_counts"
  | "residual_cap"
  | "actuation_feedback_delay_ms";

export type TargetingPipelineField =
  | "target_fov_radius_px"
  | "target_min_confidence"
  | "target_track_max_age"
  | "target_track_max_lost_age_ms"
  | "tracker_max_match_distance"
  | "tracker_position_cost_weight"
  | "tracker_iou_cost_weight"
  | "tracker_scale_cost_weight"
  | "tracker_max_size_ratio"
  | "tracker_max_association_dt_ms"
  | "tracker_kalman_acceleration_noise"
  | "tracker_kalman_measurement_noise_x"
  | "tracker_kalman_measurement_noise_y"
  | "tracker_kalman_nis_threshold"
  | "tracker_kalman_nis_hard_reject"
  | "target_class_priority"
  | "target_class_filter"
  | "target_selection_class_ratio"
  | "target_switch_min_preference_advantage"
  | "target_switch_min_continuity_score"
  | "target_switch_delay_ms"
  | "target_aim_y_ratio"
  | "target_class_aim_y_ratios"
  | "candidate_max_aspect_ratio";

export type AlgorithmParameterValues = {
  dualPhaseNearKp: number;
  dualPhaseFarKp: number;
  dualPhaseNearThreshold: number;
  dualPhaseAtanScale: number;
  actuationFeedbackDelayMs: number;
  dualPhasePredictionLeadFrames: number;
  dualPhasePredictionSmoothingFrames: number;
  dualPhasePredictionHistoryResetGapMs: number;
  velocitySpreadBasePxMs: number;
  velocitySpreadRelative: number;
  velocityChangeBasePxMs: number;
  velocityChangeRelative: number;
  dualPhasePredictionFarCapPx: number;
  dualPhasePredictionFarBaseCapPx: number;
  dualPhasePredictionFarRelativeCap: number;
  dualPhasePredictionNearCapPx: number;
  dualPhasePredictionNearBaseCapPx: number;
  dualPhasePredictionNearRelativeCap: number;
  dualPhaseNearMaxCounts: number;
  dualPhaseFarMaxCounts: number;
  dualPhaseArrivalRadiusCounts: number;
  residualCap: number;
  dualPhaseFovX: number;
  dualPhaseCountsPer360: number;
  freshnessThresholdMs: number;
};

export type AlgorithmParameterGroups = {
  responseParameters: AlgorithmNumberParameter[];
  predictionCoreParameters: AlgorithmNumberParameter[];
  predictionConfidenceParameters: AlgorithmNumberParameter[];
  predictionCapParameters: AlgorithmNumberParameter[];
  stabilityParameters: AlgorithmNumberParameter[];
  calibrationParameters: AlgorithmNumberParameter[];
};

export type TargetingParameterValues = {
  targetMinConfidence: number;
  candidateRatioMaxAspect: number;
  targetSwitchPreferenceAdvantage: number;
  targetSwitchContinuityScore: number;
  targetSwitchDelayMs: number;
  trackerMaxMatchDistance: number;
  trackerPositionCostWeight: number;
  trackerIouCostWeight: number;
  trackerScaleCostWeight: number;
  trackerMaxSizeRatio: number;
  trackerMaxAssociationDtMs: number;
  targetTrackMaxAge: number;
  targetLostGraceMs: number;
  trackerKalmanAccelerationNoise: number;
  trackerKalmanMeasurementNoiseX: number;
  trackerKalmanMeasurementNoiseY: number;
  trackerKalmanNisThreshold: number;
  trackerKalmanNisHardReject: number;
};

export type TargetingParameterGroups = {
  targetAdvancedParameters: TargetingNumberParameter[];
  trackerCoreParameters: TargetingNumberParameter[];
  trackerKalmanParameters: TargetingNumberParameter[];
};

export function buildAlgorithmParameterGroups(values: AlgorithmParameterValues): AlgorithmParameterGroups {
  return {
    responseParameters: [
      {
        key: "near_kp",
        label: "近距离响应强度（NEAR Kp）",
        detail: "目标接近准星后的主要响应参数。过高会过冲和左右往返，过低会贴近后跟不上。",
        value: values.dualPhaseNearKp,
        min: 0,
        max: 100,
        recommendedMin: 0.001,
        recommendedMax: 0.999,
        step: 0.001,
        applyMode: "save"
      },
      {
        key: "far_kp",
        label: "远距离响应强度（FAR Kp）",
        detail: "目标离准星较远时的追赶强度。它不是 KMNet 设备能力上限。",
        value: values.dualPhaseFarKp,
        min: 0,
        max: 100,
        recommendedMin: 0.001,
        recommendedMax: 0.999,
        step: 0.001,
        applyMode: "save"
      },
      {
        key: "near_threshold_px",
        label: "近远过渡位置",
        detail: "误差在该位置附近时，从 NEAR 平滑过渡到 FAR；决定多近开始进入精细控制。",
        value: values.dualPhaseNearThreshold,
        min: 0,
        max: 10000,
        recommendedMin: 0,
        recommendedMax: 1000,
        step: 0.1,
        unit: "px",
        applyMode: "save"
      },
      {
        key: "atan_scale_counts",
        label: "Atan 曲线尺度",
        detail: "决定大误差何时开始被曲线压缩。增大后中远距离输出更接近线性，减小则更早压缩。",
        value: values.dualPhaseAtanScale,
        min: 0.000001,
        max: 1000000,
        recommendedMin: 0.1,
        recommendedMax: 10000,
        step: 0.1,
        unit: "counts",
        applyMode: "save"
      }
    ],
    predictionCoreParameters: [
      {
        key: "actuation_feedback_delay_ms",
        label: "执行与反馈延迟",
        detail: "命令发出到画面可观察到响应的延迟：预测会补偿这段时间，发送后也会等待这段时间再接受新画面反馈。",
        value: values.actuationFeedbackDelayMs,
        min: 0,
        max: 1000,
        recommendedMin: 0,
        recommendedMax: 100,
        step: 0.5,
        unit: "ms",
        applyMode: "save"
      },
      {
        key: "prediction_lead_frames",
        label: "预测提前量",
        detail: "在观测帧龄和执行反馈延迟之外，沿三段速度估计额外提前多少帧。跟不上快速目标时小幅增加；急停或左右晃动过冲时降低。",
        value: values.dualPhasePredictionLeadFrames,
        min: 0,
        max: 10,
        recommendedMin: 0,
        recommendedMax: 3,
        step: 0.1,
        unit: "帧",
        applyMode: "save"
      },
      {
        key: "velocity_smoothing_frames",
        label: "三段速度平滑窗口",
        detail: "连续同向移动时平滑 3 段速度估计。增大更稳但急停、反向和 peek 响应更慢。",
        value: values.dualPhasePredictionSmoothingFrames,
        min: 0.000001,
        max: 120,
        recommendedMin: 0.1,
        recommendedMax: 12,
        step: 0.1,
        unit: "帧",
        applyMode: "save"
      },
      {
        key: "velocity_history_reset_gap_ms",
        label: "断流历史重置",
        detail: "相邻有效画面超过该时间后清空 4 点 / 3 段速度历史，避免断流后继续沿旧方向预测。",
        value: values.dualPhasePredictionHistoryResetGapMs,
        min: 0.000001,
        max: 10000,
        recommendedMin: 10,
        recommendedMax: 250,
        step: 0.1,
        unit: "ms",
        applyMode: "save"
      }
    ],
    predictionConfidenceParameters: [
      {
        key: "velocity_spread_base_px_ms",
        label: "速度离散基础容差",
        detail: "3 段速度离散程度超过基础值加相对值后，预测可信度会降低。",
        value: values.velocitySpreadBasePxMs,
        min: 0.000001,
        max: 10000,
        recommendedMin: 0.01,
        recommendedMax: 2,
        step: 0.01,
        unit: "px/ms",
        riskLevel: "advanced"
      },
      {
        key: "velocity_spread_relative",
        label: "速度离散相对容差",
        detail: "按当前速度幅度放宽离散容差，避免高速目标被固定阈值误判。",
        value: values.velocitySpreadRelative,
        min: 0,
        max: 100,
        recommendedMin: 0,
        recommendedMax: 3,
        step: 0.01,
        riskLevel: "advanced"
      },
      {
        key: "velocity_change_base_px_ms",
        label: "速度变化基础容差",
        detail: "限制相邻平滑速度的突变；超过阈值时降低预测可信度。",
        value: values.velocityChangeBasePxMs,
        min: 0.000001,
        max: 10000,
        recommendedMin: 0.01,
        recommendedMax: 2,
        step: 0.01,
        unit: "px/ms",
        riskLevel: "advanced"
      },
      {
        key: "velocity_change_relative",
        label: "速度变化相对容差",
        detail: "按已有速度幅度放宽变化阈值，适应高速但连续的运动。",
        value: values.velocityChangeRelative,
        min: 0,
        max: 100,
        recommendedMin: 0,
        recommendedMax: 3,
        step: 0.01,
        riskLevel: "advanced"
      }
    ],
    predictionCapParameters: [
      {
        key: "prediction_far_absolute_cap_px",
        label: "远距离预测硬上限",
        detail: "远距离阶段允许的最终预测位移硬上限。",
        value: values.dualPhasePredictionFarCapPx,
        min: 0,
        max: 100000,
        recommendedMin: 0,
        recommendedMax: 80,
        step: 0.1,
        unit: "px",
        riskLevel: "advanced"
      },
      {
        key: "prediction_far_base_cap_px",
        label: "远距离预测基础上限",
        detail: "动态上限的基础部分；最终仍受硬上限约束。",
        value: values.dualPhasePredictionFarBaseCapPx,
        min: 0,
        max: 100000,
        recommendedMin: 0,
        recommendedMax: 20,
        step: 0.1,
        unit: "px",
        riskLevel: "advanced"
      },
      {
        key: "prediction_far_relative_cap",
        label: "远距离预测相对上限",
        detail: "当前误差越大，允许的预测位移按该比例增加。",
        value: values.dualPhasePredictionFarRelativeCap,
        min: 0,
        max: 100,
        recommendedMin: 0,
        recommendedMax: 2,
        step: 0.01,
        riskLevel: "advanced"
      },
      {
        key: "prediction_near_absolute_cap_px",
        label: "近距离预测硬上限",
        detail: "接近准星时允许的最终预测位移硬上限。",
        value: values.dualPhasePredictionNearCapPx,
        min: 0,
        max: 100000,
        recommendedMin: 0,
        recommendedMax: 40,
        step: 0.1,
        unit: "px",
        riskLevel: "advanced"
      },
      {
        key: "prediction_near_base_cap_px",
        label: "近距离预测基础上限",
        detail: "近距离动态上限的基础部分，用于避免小误差被过量提前。",
        value: values.dualPhasePredictionNearBaseCapPx,
        min: 0,
        max: 100000,
        recommendedMin: 0,
        recommendedMax: 12,
        step: 0.1,
        unit: "px",
        riskLevel: "advanced"
      },
      {
        key: "prediction_near_relative_cap",
        label: "近距离预测相对上限",
        detail: "按当前误差比例增加近距离允许的预测位移。",
        value: values.dualPhasePredictionNearRelativeCap,
        min: 0,
        max: 100,
        recommendedMin: 0,
        recommendedMax: 2,
        step: 0.01,
        riskLevel: "advanced"
      }
    ],
    stabilityParameters: [
      {
        key: "near_max_counts_per_update",
        label: "近距离单次上限",
        detail: "靠近目标时每轮最多输出多少。降低可抑制越过瞄点，但过低会降低收敛速度。",
        value: values.dualPhaseNearMaxCounts,
        min: 1,
        max: 32767,
        recommendedMin: 1,
        recommendedMax: 2000,
        step: 1,
        unit: "counts",
        kind: "stepper",
        transform: Math.round
      },
      {
        key: "far_max_counts_per_update",
        label: "远距离单次上限",
        detail: "远距离追赶时每轮最多输出多少；它独立于 KMNet 的设备协议上限。",
        value: values.dualPhaseFarMaxCounts,
        min: 1,
        max: 32767,
        recommendedMin: 1,
        recommendedMax: 2000,
        step: 1,
        unit: "counts",
        kind: "stepper",
        transform: Math.round
      },
      {
        key: "arrival_radius_counts",
        label: "到位停止半径",
        detail: "每轴进入该范围后清空残差并停止；退出范围自动扩大 1.5 倍形成迟滞。",
        value: values.dualPhaseArrivalRadiusCounts,
        min: 0.5,
        max: 1000,
        recommendedMin: 0.5,
        recommendedMax: 50,
        step: 0.5,
        unit: "counts"
      },
      {
        key: "residual_cap",
        label: "小数残差上限",
        detail: "限制不足一个设备计数的累计余量，范围为 0～1；不是额外移动速度。",
        value: values.residualCap,
        min: 0,
        max: 1,
        step: 0.01,
        unit: "counts"
      }
    ],
    calibrationParameters: [
      {
        key: "projection_fov_x_deg",
        label: "水平视场角 FOVX",
        detail: "当前游戏水平视场角，用于把像素误差换算成角度误差。",
        value: values.dualPhaseFovX,
        min: 0.000001,
        max: 179.999999,
        recommendedMin: 30,
        recommendedMax: 179,
        step: 0.1,
        unit: "deg",
        riskLevel: "calibration"
      },
      {
        key: "projection_counts_per_360",
        label: "设备每圈 counts",
        detail: "鼠标完成 360°转向所需的真实设备计数，用于把角度需求换算成输出 counts。",
        value: values.dualPhaseCountsPer360,
        min: 0.000001,
        max: 1000000,
        recommendedMin: 1,
        recommendedMax: 100000,
        step: 1,
        unit: "counts",
        kind: "stepper",
        riskLevel: "calibration",
        transform: Math.round
      },
      {
        key: "freshness_threshold_ms",
        label: "可用观测最大帧龄",
        detail: "超过该帧龄的识别结果不会进入控制器；它是安全时效门，不是固定推理时长。",
        value: values.freshnessThresholdMs,
        min: 1,
        max: 1000,
        recommendedMin: 10,
        recommendedMax: 120,
        step: 0.1,
        unit: "ms",
        riskLevel: "calibration"
      }
    ]
  };
}

export function buildTargetingParameterGroups(values: TargetingParameterValues): TargetingParameterGroups {
  return {
    targetAdvancedParameters: [
      {
        key: "target_min_confidence",
        label: "控制目标最低置信度",
        detail: "推理结果通过模型阈值后，还必须达到该值才允许进入目标选择。",
        value: values.targetMinConfidence,
        min: 0,
        max: 1,
        step: 0.01
      },
      {
        key: "candidate_max_aspect_ratio",
        label: "候选框最大宽高比",
        detail: "拒绝宽高比或高宽比超过此值的异常细长框。值越大越宽松。",
        value: values.candidateRatioMaxAspect,
        min: 1,
        max: 100,
        recommendedMin: 1,
        recommendedMax: 20,
        step: 0.1,
        riskLevel: "advanced"
      },
      {
        key: "target_switch_min_preference_advantage",
        label: "切换最小优势",
        detail: "新候选综合分减去当前锁定目标综合分，至少达到此值才允许切换。",
        value: values.targetSwitchPreferenceAdvantage,
        min: 0,
        max: 1,
        step: 0.01
      },
      {
        key: "target_switch_min_continuity_score",
        label: "切换最小连续性",
        detail: "新候选 Track 的身份连续性至少达到此值，才允许进入切换确认。",
        value: values.targetSwitchContinuityScore,
        min: 0,
        max: 1,
        step: 0.01
      },
      {
        key: "target_switch_delay_ms",
        label: "目标切换确认延迟",
        detail: "新候选持续满足优势和连续性阈值达到此时间后，才正式替换当前目标。",
        value: values.targetSwitchDelayMs,
        min: 0,
        max: 10000,
        recommendedMin: 0,
        recommendedMax: 500,
        step: 1,
        unit: "ms",
        kind: "stepper",
        transform: Math.round
      }
    ],
    trackerCoreParameters: [
      {
        key: "tracker_max_match_distance",
        label: "归一化匹配距离",
        detail: "旧 Track 与新检测框允许关联的最大归一化距离。过大容易误关联，过小会断轨。",
        value: values.trackerMaxMatchDistance,
        min: 0.000001,
        max: 100,
        recommendedMin: 0.1,
        recommendedMax: 5,
        step: 0.05,
        riskLevel: "advanced"
      },
      {
        key: "tracker_position_cost_weight",
        label: "位置代价权重",
        detail: "目标中心位置差异在身份匹配中的权重。",
        value: values.trackerPositionCostWeight,
        min: 0,
        max: 100,
        recommendedMin: 0,
        recommendedMax: 1,
        step: 0.01,
        riskLevel: "advanced"
      },
      {
        key: "tracker_iou_cost_weight",
        label: "IoU 代价权重",
        detail: "候选框重叠程度在身份匹配中的权重。",
        value: values.trackerIouCostWeight,
        min: 0,
        max: 100,
        recommendedMin: 0,
        recommendedMax: 1,
        step: 0.01,
        riskLevel: "advanced"
      },
      {
        key: "tracker_scale_cost_weight",
        label: "尺度代价权重",
        detail: "候选框尺寸变化参与身份匹配的权重。",
        value: values.trackerScaleCostWeight,
        min: 0,
        max: 100,
        recommendedMin: 0,
        recommendedMax: 1,
        step: 0.01,
        riskLevel: "advanced"
      },
      {
        key: "tracker_max_size_ratio",
        label: "最大尺寸变化倍数",
        detail: "宽或高相对上一帧变化超过该倍数时，不允许关联为同一目标。",
        value: values.trackerMaxSizeRatio,
        min: 1,
        max: 100,
        recommendedMin: 1,
        recommendedMax: 10,
        step: 0.1,
        riskLevel: "advanced"
      },
      {
        key: "tracker_max_association_dt_ms",
        label: "最大关联时间间隔",
        detail: "两次观测间隔超过该值时，不使用旧轨迹继续关联。",
        value: values.trackerMaxAssociationDtMs,
        min: 1,
        max: 10000,
        recommendedMin: 1,
        recommendedMax: 1000,
        step: 1,
        unit: "ms",
        kind: "stepper",
        riskLevel: "advanced",
        transform: Math.round
      },
      {
        key: "target_track_max_age",
        label: "无时间戳漏检上限",
        detail: "只用于没有有效捕获时间戳的回放或降级输入；Jetson 正常主链优先使用毫秒保持时间。",
        value: values.targetTrackMaxAge,
        min: 1,
        max: 120,
        step: 1,
        unit: "帧",
        kind: "stepper",
        riskLevel: "advanced",
        transform: Math.round
      },
      {
        key: "target_track_max_lost_age_ms",
        label: "目标丢失保持",
        detail: "锁定目标短暂漏检时暂停输出并保留原身份；超过该时间后才允许其他目标接管。",
        value: values.targetLostGraceMs,
        min: 1,
        max: 10000,
        recommendedMin: 1,
        recommendedMax: 1000,
        step: 1,
        unit: "ms",
        kind: "stepper",
        riskLevel: "advanced",
        transform: Math.round
      }
    ],
    trackerKalmanParameters: [
      {
        key: "tracker_kalman_acceleration_noise",
        label: "运动响应噪声",
        detail: "越大越允许轨迹速度快速变化，但预测协方差也会更快增长；它只影响身份关联。",
        value: values.trackerKalmanAccelerationNoise,
        min: 0.000001,
        max: 1000000,
        recommendedMin: 0.001,
        recommendedMax: 100000,
        step: 10,
        riskLevel: "advanced"
      },
      {
        key: "tracker_kalman_measurement_noise_x",
        label: "X 轴观测噪声",
        detail: "检测框中心在 X 轴的预期方差。增大后更容忍横向框抖动，但关联门会相应变宽。",
        value: values.trackerKalmanMeasurementNoiseX,
        min: 0.000001,
        max: 100000,
        recommendedMin: 0.001,
        recommendedMax: 1000,
        step: 0.5,
        unit: "px²",
        riskLevel: "advanced"
      },
      {
        key: "tracker_kalman_measurement_noise_y",
        label: "Y 轴观测噪声",
        detail: "检测框中心在 Y 轴的预期方差。Y 轴框高变化明显时可以独立调整。",
        value: values.trackerKalmanMeasurementNoiseY,
        min: 0.000001,
        max: 100000,
        recommendedMin: 0.001,
        recommendedMax: 1000,
        step: 0.5,
        unit: "px²",
        riskLevel: "advanced"
      },
      {
        key: "tracker_kalman_nis_threshold",
        label: "NIS 可信阈值",
        detail: "创新量低于该值时卡尔曼状态可作为可信关联预测；必须不高于硬拒绝阈值。",
        value: values.trackerKalmanNisThreshold,
        min: 0.000001,
        max: values.trackerKalmanNisHardReject,
        recommendedMin: 0.001,
        recommendedMax: Math.min(100, values.trackerKalmanNisHardReject),
        step: 0.1,
        riskLevel: "advanced"
      },
      {
        key: "tracker_kalman_nis_hard_reject",
        label: "NIS 硬拒绝阈值",
        detail: "创新量超过该值时，该检测与旧 Track 不允许关联。降低会减少误关联，过低会造成频繁断轨。",
        value: values.trackerKalmanNisHardReject,
        min: Math.max(0.000001, values.trackerKalmanNisThreshold),
        max: 1000000,
        recommendedMin: Math.max(0.001, values.trackerKalmanNisThreshold),
        recommendedMax: 100,
        step: 0.1,
        riskLevel: "advanced"
      }
    ]
  };
}
