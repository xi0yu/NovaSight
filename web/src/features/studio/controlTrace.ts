import type { RuntimeOutputTraceState } from "../../api";

export type ControlTraceState = "ready" | "blocked" | "waiting" | "idle";

export type ControlTraceStepId =
  | "target"
  | "trigger"
  | "aim"
  | "controller"
  | "recoil"
  | "limiter"
  | "output";

export type ControlTraceStep = {
  id: ControlTraceStepId;
  label: string;
  state: ControlTraceState;
  stateLabel?: string;
  value: string;
  detail: string;
  evidence: string;
};

export type ControlTraceSummary = {
  state: ControlTraceState;
  title: string;
  completed: number;
  total: number;
  steps: ControlTraceStep[];
};

export type BuildControlTraceInput = {
  runtimeRunning: boolean;
  detectionBatchFps: number | null;
  publishedBatches: number | null;
  consumedBatches: number | null;
  targetingBatches: number | null;
  detectionDataAgeMs: number | null;
  freshnessThresholdMs: number | null;
  detectionCount: number | null;
  rawCandidateCount: number | null;
  eligibleCandidateCount: number | null;
  selectedTargetCount: number | null;
  targetPipelineStage: string;
  targetPipelineCode: string;
  targetPipelineMessage: string;
  targetPipelineRejections: string;
  hasTarget: boolean;
  trackId: number | null;
  classLabel: string;
  targetScore: number | null;
  observedAim: string;
  controlAim: string;
  controlCenter: string;
  controlError: string;
  errorDistancePx: number | null;
  controlFrameAgeMs: number | null;
  measurementDtMs: number | null;
  predictionEnabled: boolean;
  predictionAllowedX?: boolean | null;
  predictionAllowedY?: boolean | null;
  predictionSafeOffset?: string;
  controllerActive: boolean;
  controllerMode: string;
  movementStrategy: string;
  fullError: string;
  floatDemand: string;
  maxOutputXCounts?: number | null;
  maxOutputYCounts?: number | null;
  integerCommand: string;
  hardwareTriggerRequired: boolean;
  fireDelayEnabled: boolean;
  fireDelayConfiguredMs: number;
  fireDelayPending: boolean;
  fireDelayElapsedMs: number | null;
  fireDelayRemainingMs: number | null;
  recoilEnabled?: boolean;
  recoilState?: string;
  recoilStatus?: string;
  recoilRemainingMs?: number | null;
  recoilRequestedY?: number | null;
  recoilEmittedY?: number | null;
  outputEnabled: boolean;
  willEmit: boolean | null;
  triggerActive: boolean | null;
  noSendReason: string;
  kmnetRuntimeConnected: boolean;
  kmnetConnectionLabel: string;
  deviceLastError: string;
  outputTrace: RuntimeOutputTraceState | null;
};

const OUTPUT_TRACE_LABELS: Record<string, string> = {
  runtime_stopped: "主链未运行",
  inference_not_running: "推理未运行",
  no_detection_batches: "等待识别结果",
  runtime_not_consuming_batches: "控制链路未读取",
  stale_detection_batch: "检测已过期",
  target_not_selected: "未选中目标",
  control_not_calculated: "控制未计算",
  hardware_output_disabled: "硬件输出不可用",
  output_gate_closed: "输出门关闭",
  trigger_inactive: "等待触发",
  device_not_connected: "设备未连接",
  control_blocked: "控制阻断",
  generation_fenced: "等待新配置帧",
  device_send_failed: "设备发送失败",
  device_output_disabled: "设备输出已停用",
  TIMESTAMP_DOMAIN_INVALID: "时间戳域无效",
  STALE_OBSERVATION: "控制观测已过期",
  NON_MONOTONIC_OBSERVATION: "控制时间戳非单调",
  TARGET_INVALID: "控制目标无效",
  GEOMETRY_INVALID: "控制几何无效",
  TRIGGER_INACTIVE: "等待触发",
  TRIGGER_DELAY_PENDING: "等待开火延迟",
  DEAD_ZONE: "目标位于死区",
  DEMAND_OUT_OF_RANGE: "控制需求越界",
  ready: "输出链路已贯通"
};

function outputTraceLabel(trace: RuntimeOutputTraceState): string {
  return OUTPUT_TRACE_LABELS[trace.code] ?? trace.code;
}

function outputTraceState(trace: RuntimeOutputTraceState): ControlTraceState {
  if (trace.state === "ready") {
    return "ready";
  }
  if (trace.state === "blocked") {
    return "blocked";
  }
  if (trace.state === "idle") {
    return "idle";
  }
  return "waiting";
}

function positive(value: number | null): boolean {
  return value !== null && value > 0;
}

function present(value: string): boolean {
  return value.trim() !== "" && value.trim() !== "—" && value.trim() !== "-";
}

function formatInteger(value: number | null): string {
  return value === null ? "—" : String(Math.trunc(value));
}

function formatNumber(value: number | null, digits = 2, unit = ""): string {
  if (value === null) {
    return "—";
  }
  const rendered = value.toFixed(digits);
  return unit ? `${rendered} ${unit}` : rendered;
}

function freshState(ageMs: number | null, thresholdMs: number | null): "fresh" | "stale" | "unknown" {
  if (ageMs === null || thresholdMs === null) {
    return "unknown";
  }
  return ageMs <= thresholdMs ? "fresh" : "stale";
}

function buildTargetStep(input: BuildControlTraceInput): ControlTraceStep {
  const freshness = freshState(input.detectionDataAgeMs, input.freshnessThresholdMs);
  const hasRuntimeInput = positive(input.consumedBatches) || positive(input.targetingBatches);
  const hasPublished = positive(input.publishedBatches);
  if (freshness === "stale" && input.runtimeRunning) {
    return {
      id: "target",
      label: "目标输入",
      state: "blocked",
      value: formatNumber(input.detectionDataAgeMs, 2, "ms"),
      detail: "最新识别结果超过控制新鲜度阈值，目标选择不会把过期样本交给控制算法。",
      evidence: `published=${formatInteger(input.publishedBatches)} · consumed=${formatInteger(input.consumedBatches)} · targeting=${formatInteger(input.targetingBatches)}`
    };
  }
  if (input.hasTarget) {
    return {
      id: "target",
      label: "目标输入",
      state: "ready",
      value: input.trackId === null ? "已选择" : `track ${Math.trunc(input.trackId)}`,
      detail: input.classLabel ? `${input.classLabel} 已作为主要目标进入控制链。` : "主要目标已进入控制链。",
      evidence: `fps=${formatNumber(input.detectionBatchFps, 1)} · 候选/合格/选择=${formatInteger(input.rawCandidateCount)}/${formatInteger(input.eligibleCandidateCount)}/${formatInteger(input.selectedTargetCount)} · score=${formatNumber(input.targetScore, 3)}`
    };
  }
  const hasDiagnostics = input.targetPipelineStage || input.targetPipelineCode || input.targetPipelineMessage;
  return {
    id: "target",
    label: "目标输入",
    state: hasDiagnostics
      ? "blocked"
      : hasRuntimeInput || hasPublished
        ? "waiting"
        : input.runtimeRunning ? "waiting" : "idle",
    value: formatInteger(input.selectedTargetCount),
    detail: input.targetPipelineMessage || (input.runtimeRunning ? "等待选择主要目标。" : "主链未运行，暂未产生目标输入。"),
    evidence: input.targetPipelineRejections || input.targetPipelineCode || `detections=${formatInteger(input.detectionCount)} · published=${formatInteger(input.publishedBatches)}`
  };
}

function buildAimStep(input: BuildControlTraceInput): ControlTraceStep {
  const stale = freshState(input.controlFrameAgeMs, input.freshnessThresholdMs) === "stale";
  if (stale) {
    return {
      id: "aim",
      label: "目标速度预测",
      state: "blocked",
      value: formatNumber(input.controlFrameAgeMs, 2, "ms"),
      detail: "当前控制观测过期，不能用于物理输出。",
      evidence: `center=${input.controlCenter} · aim=${input.controlAim}`
    };
  }
  if (input.hasTarget && present(input.controlAim) && present(input.controlError)) {
    const predictionTelemetryAvailable =
      input.predictionAllowedX !== undefined || input.predictionAllowedY !== undefined;
    const predictionApplied = input.predictionAllowedX === true || input.predictionAllowedY === true;
    const predictionDetail = !input.predictionEnabled
      ? "预测已关闭，当前观测直接进入控制误差。"
      : predictionTelemetryAvailable && predictionApplied
        ? "本次运行样本通过预测门控，安全预测位移已进入控制误差。"
        : predictionTelemetryAvailable
          ? "本次运行样本未通过预测门控，当前观测直接进入控制误差。"
          : "等待运行样本报告预测门控状态。";
    return {
      id: "aim",
      label: "目标速度预测",
      state: "ready",
      value: input.predictionEnabled
        ? predictionApplied ? "已介入" : "观测直入"
        : "已关闭",
      detail: predictionDetail,
      evidence: `allowed=${String(input.predictionAllowedX ?? "—")}/${String(input.predictionAllowedY ?? "—")} · offset=${input.predictionSafeOffset || "—"}`
    };
  }
  return {
    id: "aim",
    label: "目标速度预测",
    state: input.hasTarget ? "waiting" : "idle",
    value: "—",
    detail: input.hasTarget ? "等待控制瞄点与画面中心。" : "没有目标时不计算控制误差。",
    evidence: `frame_age=${formatNumber(input.controlFrameAgeMs, 2, "ms")} · dt=${formatNumber(input.measurementDtMs, 3, "ms")}`
  };
}

function buildLimiterStep(input: BuildControlTraceInput): ControlTraceStep {
  if (!input.controllerActive) {
    return {
      id: "limiter",
      label: "X/Y 输出限幅",
      state: input.hasTarget ? "waiting" : "idle",
      value: "—",
      detail: input.hasTarget ? "等待连续控制器产生浮点需求。" : "没有控制需求时限幅器保持空闲。",
      evidence: `limit_x/y=${formatNumber(input.maxOutputXCounts ?? null, 0, "counts")}/${formatNumber(input.maxOutputYCounts ?? null, 0, "counts")}`
    };
  }
  const maxX = input.maxOutputXCounts ?? null;
  const maxY = input.maxOutputYCounts ?? null;
  const limitValue = `±${formatInteger(maxX)} / ±${formatInteger(maxY)}`;
  const limitEvidence = `tracking=${input.integerCommand} · recoil_y=${formatInteger(input.recoilRequestedY ?? null)} · limit_x/y=${formatNumber(maxX, 0, "counts")}/${formatNumber(maxY, 0, "counts")}`;
  if (maxX === null || maxY === null) {
    return {
      id: "limiter",
      label: "固定 X/Y 限幅",
      state: "waiting",
      value: limitValue,
      detail: "等待当前运行配置报告完整的 X/Y 固定输出上限。",
      evidence: limitEvidence
    };
  }
  if (!input.outputEnabled) {
    return {
      id: "limiter",
      label: "固定 X/Y 限幅",
      state: "idle",
      stateLabel: "未执行",
      value: limitValue,
      detail: "固定限幅规则已配置，但输出门关闭，当前控制意图不会进入设备边界。",
      evidence: limitEvidence
    };
  }
  if ((input.hardwareTriggerRequired && input.triggerActive !== true) || !input.kmnetRuntimeConnected) {
    return {
      id: "limiter",
      label: "固定 X/Y 限幅",
      state: "waiting",
      value: limitValue,
      detail: input.hardwareTriggerRequired && input.triggerActive !== true
        ? "等待触发条件；设备 worker 尚未执行压枪叠加与最终 clamp。"
        : "等待设备通道；设备 worker 尚未执行压枪叠加与最终 clamp。",
      evidence: limitEvidence
    };
  }
  if (input.willEmit !== true) {
    return {
      id: "limiter",
      label: "固定 X/Y 限幅",
      state: "blocked",
      value: limitValue,
      detail: input.noSendReason || "当前控制样本未获准进入设备边界。",
      evidence: limitEvidence
    };
  }
  return {
    id: "limiter",
    label: "固定 X/Y 限幅",
    state: "ready",
    stateLabel: "边界就绪",
    value: limitValue,
    detail: "当前条件允许控制意图进入设备 worker；worker 会先叠加压枪，再执行固定轴向 clamp。",
    evidence: limitEvidence
  };
}

function buildTriggerDelayStep(input: BuildControlTraceInput): ControlTraceStep {
  if (!input.hardwareTriggerRequired) {
    return {
      id: "trigger",
      label: "触发与开火延迟",
      state: "ready",
      value: "直接触发",
      detail: "直接触发模式不等待鼠标按键持续时间。",
      evidence: "trigger_mode=always"
    };
  }
  if (input.triggerActive !== true) {
    return {
      id: "trigger",
      label: "触发与开火延迟",
      state: "waiting",
      value: "等待按键",
      detail: "鼠标按键未按下，控制算法保持未启动。",
      evidence: "trigger_active=false"
    };
  }
  if (input.fireDelayPending) {
    return {
      id: "trigger",
      label: "触发与开火延迟",
      state: "waiting",
      value: "持续按住",
      detail: "按住时间尚未超过开火延迟，预测和控制算法均未执行。",
      evidence: `elapsed=${formatNumber(input.fireDelayElapsedMs, 2, "ms")} · remaining=${formatNumber(input.fireDelayRemainingMs, 2, "ms")}`
    };
  }
  return {
    id: "trigger",
    label: "触发与开火延迟",
    state: "ready",
    value: input.fireDelayEnabled ? "门槛已通过" : "立即启动",
    detail: input.fireDelayEnabled
      ? "按键持续时间已超过门槛，当前最新目标可以进入控制算法。"
      : "开火延迟已关闭，按键按下后立即启动控制算法。",
    evidence: `configured=${formatNumber(input.fireDelayConfiguredMs, 0, "ms")}`
  };
}

function buildRecoilStep(input: BuildControlTraceInput): ControlTraceStep {
  if (!input.outputEnabled) {
    return {
      id: "recoil",
      label: "压枪叠加",
      state: "idle",
      value: "未计算",
      detail: "物理输出门关闭，设备边界不会推进压枪叠加节拍。",
      evidence: "output_enabled=false"
    };
  }
  if (input.triggerActive === false || !input.kmnetRuntimeConnected) {
    return {
      id: "recoil",
      label: "压枪叠加",
      state: "waiting",
      value: "等待输出条件",
      detail: input.triggerActive === false
        ? "硬件触发条件未激活，设备边界不会计算本轮压枪叠加。"
        : "设备通道未连接，设备边界不会计算本轮压枪叠加。",
      evidence: `trigger=${String(input.triggerActive ?? "—")} · connected=${String(input.kmnetRuntimeConnected)}`
    };
  }
  if (!input.recoilEnabled) {
    return {
      id: "recoil",
      label: "压枪叠加",
      state: "ready",
      value: "关闭（旁路）",
      detail: "独立压枪未启用，已准入的跟踪命令在设备边界保持不变。",
      evidence: "recoil_enabled=false"
    };
  }
  if (!input.controllerActive || !input.recoilState) {
    return {
      id: "recoil",
      label: "压枪叠加",
      state: input.hasTarget ? "waiting" : "idle",
      value: "等待样本",
      detail: "等待设备边界报告真实左键和压枪叠加结果。",
      evidence: `remaining=${formatNumber(input.recoilRemainingMs ?? null, 2, "ms")}`
    };
  }
  return {
    id: "recoil",
    label: "压枪叠加",
    state: "ready",
    value: input.recoilStatus || input.recoilState,
    detail: "压枪只在控制算法启动后按独立间隔叠加 +Y。",
    evidence: `state=${input.recoilState} · remaining=${formatNumber(input.recoilRemainingMs ?? null, 2, "ms")} · requested/emitted=${formatInteger(input.recoilRequestedY ?? null)}/${formatInteger(input.recoilEmittedY ?? null)}`
  };
}

function buildControllerStep(input: BuildControlTraceInput): ControlTraceStep {
  if (input.controllerActive) {
    return {
      id: "controller",
      label: "连续非线性控制",
      state: "ready",
      value: present(input.integerCommand) ? input.integerCommand : "已计算",
      detail: `${input.controllerMode}${input.movementStrategy ? ` · ${input.movementStrategy}` : ""}`,
      evidence: `full=${input.fullError} · atan=${input.floatDemand}`
    };
  }
  return {
    id: "controller",
    label: "连续非线性控制",
    state: input.hasTarget ? "waiting" : "idle",
    value: "—",
    detail: input.hasTarget ? "等待目标样本进入连续 Atan 控制器。" : "没有目标时控制器保持空闲。",
    evidence: input.noSendReason || "waiting"
  };
}

function buildOutputStep(input: BuildControlTraceInput): ControlTraceStep {
  if (!input.outputEnabled) {
    return {
      id: "output",
      label: "命令输出",
      state: "idle",
      value: "暂停",
      detail: "物理输出关闭，不会发送鼠标偏移。",
      evidence: input.noSendReason || "output_enabled=false"
    };
  }
  if (input.willEmit === true && !input.kmnetRuntimeConnected) {
    return {
      id: "output",
      label: "命令输出",
      state: "blocked",
      value: "设备未连接",
      detail: "当前命令已经通过控制门，但 kmNet 运行时通道未连接。",
      evidence: input.deviceLastError || input.kmnetConnectionLabel
    };
  }
  if (input.kmnetRuntimeConnected && input.willEmit === true) {
    return {
      id: "output",
      label: "命令输出",
      state: "ready",
      stateLabel: "已准入",
      value: "等待设备结果",
      detail: "当前控制意图已通过触发、输出门与设备连接检查；设备回执只在折叠诊断中展示。",
      evidence: `trigger=${String(input.triggerActive ?? "—")} · connected=true`
    };
  }
  if (input.kmnetRuntimeConnected) {
    return {
      id: "output",
      label: "命令输出",
      state: "waiting",
      value: "已连接",
      detail: input.noSendReason || "设备通道已连接，等待第一条被接受的控制命令。",
      evidence: `trigger=${String(input.triggerActive ?? "—")} · ${input.kmnetConnectionLabel}`
    };
  }
  const waitingForTrigger = input.noSendReason.includes("触发") || input.triggerActive === false;
  return {
    id: "output",
    label: "命令输出",
    state: waitingForTrigger || !input.hasTarget ? "waiting" : "blocked",
    value: "未发送",
    detail: input.noSendReason || "等待输出门准入和 kmNet 运行时连接。",
    evidence: input.deviceLastError || input.kmnetConnectionLabel
  };
}

export function buildControlTrace(input: BuildControlTraceInput): ControlTraceSummary {
  const steps = [
    buildTargetStep(input),
    buildTriggerDelayStep(input),
    buildAimStep(input),
    buildControllerStep(input),
    buildRecoilStep(input),
    buildLimiterStep(input),
    buildOutputStep(input)
  ];
  const completed = steps.filter((step) => step.state === "ready").length;
  const hasBlocked = steps.some((step) => step.state === "blocked");
  const hasReadyController = steps.some((step) => step.id === "controller" && step.state === "ready");
  const derivedState: ControlTraceState = hasBlocked
    ? "blocked"
    : completed === steps.length
      ? "ready"
      : hasReadyController
        ? "waiting"
        : input.runtimeRunning ? "waiting" : "idle";
  const traceState = input.outputTrace ? outputTraceState(input.outputTrace) : null;
  const state = traceState && traceState !== "ready" ? traceState : derivedState;
  const title = state === "ready"
    ? "控制链路已具备输出条件"
    : state === "blocked"
      ? input.outputTrace
        ? outputTraceLabel(input.outputTrace)
        : "控制链路存在阻断"
      : input.runtimeRunning
        ? input.outputTrace
          ? outputTraceLabel(input.outputTrace)
        : "控制链路正在等待实时状态"
        : "控制链路等待主链启动";
  return {
    state,
    title,
    completed,
    total: steps.length,
    steps
  };
}
