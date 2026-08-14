import type { ConfigFieldSchema, ConfigSchemaResponse, RuntimeConfigValue } from "../../api";
import type {
  ParameterApplyMode,
  ParameterNumberKind,
  ParameterRiskLevel
} from "./StudioControls";

export type StudioNumberParameter<Field extends string> = {
  key: Field;
  label: string;
  detail: string;
  formula?: string;
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

export type AlgorithmNumberParameter = StudioNumberParameter<ControlPipelineField>;
export type TargetingNumberParameter = StudioNumberParameter<TargetingPipelineField>;
export type AlgorithmSettingsSection = "response" | "prediction" | "calibration";

export const CONTROL_PIPELINE_FIELDS = [
  "freshness_threshold_ms",
  "projection_fov_x_deg",
  "projection_counts_per_360",
  "p_response_scale",
  "p_response_boost",
  "p_response_curve_shape",
  "max_output_x_counts",
  "max_output_y_counts",
  "prediction_enabled",
  "velocity_history_reset_gap_ms",
  "prediction_lead_ms",
  "prediction_cap_px",
  "actuation_feedback_delay_ms"
] as const;

export type ControlPipelineField = typeof CONTROL_PIPELINE_FIELDS[number];

export const TARGETING_PIPELINE_FIELDS = [
  "target_fov_radius_px",
  "target_min_confidence",
  "target_track_max_age",
  "target_track_max_lost_age_ms",
  "tracker_max_match_distance",
  "tracker_position_cost_weight",
  "tracker_iou_cost_weight",
  "tracker_scale_cost_weight",
  "tracker_max_size_ratio",
  "tracker_max_association_dt_ms",
  "tracker_kalman_acceleration_noise",
  "tracker_kalman_measurement_noise_x",
  "tracker_kalman_measurement_noise_y",
  "tracker_kalman_max_predict_dt_ms",
  "tracker_kalman_max_predict_missing_ms",
  "tracker_kalman_max_predict_steps",
  "tracker_kalman_nis_threshold",
  "tracker_kalman_nis_hard_reject",
  "target_class_priority",
  "target_class_filter",
  "target_selection_class_ratio",
  "target_switch_min_preference_advantage",
  "target_switch_min_continuity_score",
  "target_switch_delay_ms",
  "target_aim_y_ratio",
  "target_class_aim_y_ratios",
  "candidate_max_aspect_ratio"
] as const;

export type TargetingPipelineField = typeof TARGETING_PIPELINE_FIELDS[number];

export type ConfigFieldIndex = ReadonlyMap<string, ConfigFieldSchema>;

export type StudioConfigSchemaIssue = {
  path: string;
  reason: string;
};

const NUMERIC_SCHEMA_TYPES = new Set<ConfigFieldSchema["type"]>(["int", "float"]);

function parameterApplyMode(field: ConfigFieldSchema): ParameterApplyMode {
  if (field.apply_mode === "hot_update") {
    return "live";
  }
  if (field.apply_mode === "epoch_reload") {
    return "reload";
  }
  return "restart";
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function pipelinePath(field: string): string {
  return `pipeline.${field}`;
}

export function buildConfigFieldIndex(schema: ConfigSchemaResponse | null | undefined): ConfigFieldIndex | null {
  if (!schema) {
    return null;
  }
  const fields = new Map<string, ConfigFieldSchema>();
  for (const section of schema.sections) {
    for (const field of section.fields) {
      fields.set(field.path, field);
    }
  }
  return fields;
}

export function validateStudioConfigSchema(schema: ConfigSchemaResponse): StudioConfigSchemaIssue[] {
  const fields = buildConfigFieldIndex(schema);
  if (!fields) {
    return [{ path: "schema", reason: "missing config schema" }];
  }
  const issues: StudioConfigSchemaIssue[] = [];
  if (!schema.algorithm) {
    issues.push({ path: "algorithm", reason: "backend schema must expose the active control algorithm descriptor" });
  } else {
    if (!isFiniteNumber(schema.algorithm.response.atan_scale_counts) || schema.algorithm.response.atan_scale_counts <= 0) {
      issues.push({ path: "algorithm.response.atan_scale_counts", reason: "Atan scale must be a positive number" });
    }
    if (schema.algorithm.prediction.aim_history_points !== 4 || schema.algorithm.prediction.velocity_segments !== 3) {
      issues.push({ path: "algorithm.prediction", reason: "Studio prediction display expects 4 aim points and 3 velocity segments" });
    }
  }
  for (const key of CONTROL_PIPELINE_FIELDS) {
    const path = pipelinePath(key);
    const field = fields.get(path);
    if (!field) {
      issues.push({ path, reason: "Studio control parameter is not exposed by backend schema" });
      continue;
    }
    if (key === "prediction_enabled") {
      if (field.type !== "bool") {
        issues.push({ path, reason: `expected bool schema field, got ${field.type}` });
      }
    } else if (!NUMERIC_SCHEMA_TYPES.has(field.type)) {
      issues.push({ path, reason: `expected numeric schema field, got ${field.type}` });
    }
    if (field.apply_mode !== "hot_update") {
      issues.push({ path, reason: "control algorithm field must be hot-applied by the backend" });
    }
  }
  for (const key of TARGETING_PIPELINE_FIELDS) {
    const path = pipelinePath(key);
    const field = fields.get(path);
    if (!field) {
      issues.push({ path, reason: "Studio targeting parameter is not exposed by backend schema" });
      continue;
    }
    if (field.apply_mode !== "hot_update") {
      issues.push({ path, reason: "targeting field must be hot-applied by the backend" });
    }
  }
  return issues;
}

function schemaBackedNumberParameter<Field extends string>(
  schemaIndex: ConfigFieldIndex | null | undefined,
  section: "pipeline",
  parameter: StudioNumberParameter<Field>
): StudioNumberParameter<Field> | null {
  if (!schemaIndex) {
    return parameter;
  }
  const field = schemaIndex.get(`${section}.${parameter.key}`);
  if (!field || !NUMERIC_SCHEMA_TYPES.has(field.type)) {
    return null;
  }
  const min = isFiniteNumber(field.min) ? field.min : parameter.min;
  const max = isFiniteNumber(field.max) ? field.max : parameter.max;
  return {
    ...parameter,
    label: field.label || parameter.label,
    min,
    max,
    unit: field.unit ?? parameter.unit,
    applyMode: parameterApplyMode(field)
  };
}

function schemaBackedNumberParameters<Field extends string>(
  schemaIndex: ConfigFieldIndex | null | undefined,
  parameters: StudioNumberParameter<Field>[]
): StudioNumberParameter<Field>[] {
  return parameters
    .map((parameter) => schemaBackedNumberParameter(schemaIndex, "pipeline", parameter))
    .filter((parameter): parameter is StudioNumberParameter<Field> => parameter !== null);
}

export type AlgorithmParameterValues = {
  pResponseScale: number;
  pResponseBoost: number;
  pResponseCurveShape: number;
  actuationFeedbackDelayMs: number;
  controlPredictionLeadMs: number;
  controlPredictionHistoryResetGapMs: number;
  controlPredictionCapPx: number;
  controlFovX: number;
  controlCountsPer360: number;
  freshnessThresholdMs: number;
};

export type AlgorithmParameterGroups = {
  responseParameters: AlgorithmNumberParameter[];
  predictionCoreParameters: AlgorithmNumberParameter[];
  predictionCapParameters: AlgorithmNumberParameter[];
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
  trackerKalmanMaxPredictDtMs: number;
  trackerKalmanMaxPredictMissingMs: number;
  trackerKalmanMaxPredictSteps: number;
  trackerKalmanNisThreshold: number;
  trackerKalmanNisHardReject: number;
};

export type TargetingParameterGroups = {
  targetAdvancedParameters: TargetingNumberParameter[];
  trackerCoreParameters: TargetingNumberParameter[];
  trackerKalmanParameters: TargetingNumberParameter[];
};

export function buildAlgorithmParameterGroups(
  values: AlgorithmParameterValues,
  schemaIndex?: ConfigFieldIndex | null
): AlgorithmParameterGroups {
  const groups: AlgorithmParameterGroups = {
    responseParameters: [
      {
        key: "p_response_scale",
        label: "基础响应 K_base",
        formula: "K_base / p_response_scale",
        detail: "写入 pipeline.p_response_scale，直接乘在 Atan 输出前。觉得整体慢但轨迹稳定时先小幅提高它；过高会让所有误差区间一起变冲。",
        value: values.pResponseScale,
        min: 0,
        max: 100,
        recommendedMin: 0.001,
        recommendedMax: 1,
        step: 0.001,
        applyMode: "live"
      },
      {
        key: "p_response_boost",
        label: "动态增强 B",
        formula: "B / p_response_boost",
        detail: "写入 pipeline.p_response_boost，控制 R(r) 随误差距离最多额外放大多少。远距离响应不足时小幅提高；过高会让中远距离变冲。",
        value: values.pResponseBoost,
        min: 0,
        max: 100,
        recommendedMin: 0,
        recommendedMax: 2,
        step: 0.001,
        applyMode: "live"
      },
      {
        key: "p_response_curve_shape",
        label: "过渡形状 gamma",
        formula: "gamma / p_response_curve_shape",
        detail: "写入 pipeline.p_response_curve_shape，只改变增强随误差半径 r 进入的早晚。低于 1 更早变强，高于 1 更晚变强；它不是整体速度旋钮。",
        value: values.pResponseCurveShape,
        min: 0.5,
        max: 4,
        recommendedMin: 0.5,
        recommendedMax: 2,
        step: 0.01,
        applyMode: "live"
      }
    ],
    predictionCoreParameters: [
      {
        key: "prediction_lead_ms",
        label: "预测提前量",
        formula: "T_extra",
        detail: "在观测帧龄和执行反馈延迟之外，沿 aim 点速度额外提前的时间。跟不上移动目标时小幅增加；急停或过度预判时降低。",
        value: values.controlPredictionLeadMs,
        min: 0,
        max: 1000,
        recommendedMin: 0,
        recommendedMax: 80,
        step: 0.5,
        unit: "ms",
        applyMode: "live"
      },
      {
        key: "velocity_history_reset_gap_ms",
        label: "断流历史重置",
        formula: "history",
        detail: "相邻有效画面超过该时间后清空 4 点 / 3 段速度历史，避免断流后继续沿上次方向预测。",
        value: values.controlPredictionHistoryResetGapMs,
        min: 0.000001,
        max: 10000,
        recommendedMin: 10,
        recommendedMax: 250,
        step: 0.1,
        unit: "ms",
        applyMode: "live"
      }
    ],
    predictionCapParameters: [
      {
        key: "prediction_cap_px",
        label: "预测位移上限",
        formula: "cap_pred",
        detail: "目标速度预测最多把 aim 点向未来推进多少像素。这是预测位移 cap，不是鼠标输出上限。",
        value: values.controlPredictionCapPx,
        min: 0,
        max: 100000,
        recommendedMin: 0,
        recommendedMax: 80,
        step: 0.1,
        unit: "px"
      }
    ],
    calibrationParameters: [
      {
        key: "actuation_feedback_delay_ms",
        label: "预测执行延迟",
        formula: "T_delay",
        detail: "命令发出到画面可观察到响应的系统延迟，只参与目标速度预测时域。",
        value: values.actuationFeedbackDelayMs,
        min: 0,
        max: 1000,
        recommendedMin: 0,
        recommendedMax: 100,
        step: 0.5,
        unit: "ms",
        applyMode: "live",
        riskLevel: "calibration"
      },
      {
        key: "projection_fov_x_deg",
        label: "水平视场角 FOVX",
        formula: "proj_fov",
        detail: "当前游戏水平视场角，用于把像素误差换算成角度误差。",
        value: values.controlFovX,
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
        formula: "proj_counts",
        detail: "鼠标完成 360°转向所需的真实设备计数，用于把角度需求换算成输出 counts。",
        value: values.controlCountsPer360,
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
        formula: "freshness",
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
  return {
    responseParameters: schemaBackedNumberParameters(schemaIndex, groups.responseParameters),
    predictionCoreParameters: schemaBackedNumberParameters(schemaIndex, groups.predictionCoreParameters),
    predictionCapParameters: schemaBackedNumberParameters(schemaIndex, groups.predictionCapParameters),
    calibrationParameters: schemaBackedNumberParameters(schemaIndex, groups.calibrationParameters)
  };
}

export function buildTargetingParameterGroups(
  values: TargetingParameterValues,
  schemaIndex?: ConfigFieldIndex | null
): TargetingParameterGroups {
  const groups: TargetingParameterGroups = {
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
        detail: "已有 Track 与新检测框允许关联的最大归一化距离。过大容易误关联，过小会断轨。",
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
        detail: "两次观测间隔超过该值时，不使用上一轨迹继续关联。",
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
        key: "tracker_kalman_max_predict_dt_ms",
        label: "单步预测上限",
        detail: "相邻两次关联预测最多按多少毫秒推进；限制异常长帧间隔把卡尔曼轨迹外推过远。",
        value: values.trackerKalmanMaxPredictDtMs,
        min: 1,
        max: 1000,
        recommendedMin: 5,
        recommendedMax: 80,
        step: 1,
        unit: "ms",
        kind: "stepper",
        riskLevel: "advanced",
        transform: Math.round
      },
      {
        key: "tracker_kalman_max_predict_missing_ms",
        label: "丢失预测窗口",
        detail: "目标短暂漏检时，卡尔曼状态最多保留多久用于身份关联；它不直接输出鼠标提前量。",
        value: values.trackerKalmanMaxPredictMissingMs,
        min: 1,
        max: 10000,
        recommendedMin: 10,
        recommendedMax: 250,
        step: 1,
        unit: "ms",
        kind: "stepper",
        riskLevel: "advanced",
        transform: Math.round
      },
      {
        key: "tracker_kalman_max_predict_steps",
        label: "连续预测步数",
        detail: "没有新观测时最多允许连续外推多少次。调高更能跨短暂漏检，过高会增加误关联风险。",
        value: values.trackerKalmanMaxPredictSteps,
        min: 0,
        max: 120,
        recommendedMin: 0,
        recommendedMax: 12,
        step: 1,
        unit: "步",
        kind: "stepper",
        riskLevel: "advanced",
        transform: Math.round
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
        detail: "创新量超过该值时，该检测与已有 Track 不允许关联。降低会减少误关联，过低会造成频繁断轨。",
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
  return {
    targetAdvancedParameters: schemaBackedNumberParameters(schemaIndex, groups.targetAdvancedParameters),
    trackerCoreParameters: schemaBackedNumberParameters(schemaIndex, groups.trackerCoreParameters),
    trackerKalmanParameters: schemaBackedNumberParameters(schemaIndex, groups.trackerKalmanParameters)
  };
}
