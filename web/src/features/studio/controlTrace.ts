import type { RuntimeOutputTraceState } from "../../api";

export type ControlTraceState = "ready" | "blocked" | "waiting" | "idle";

export type ControlTraceStepId =
  | "batch"
  | "target"
  | "aim"
  | "controller"
  | "limiter"
  | "recoil"
  | "gate"
  | "device";

export type ControlTraceStep = {
  id: ControlTraceStepId;
  label: string;
  state: ControlTraceState;
  value: string;
  detail: string;
  evidence: string;
};

export type ControlTraceFact = {
  label: string;
  value: string;
  detail: string;
};

export type ControlTraceSummary = {
  state: ControlTraceState;
  title: string;
  detail: string;
  completed: number;
  total: number;
  steps: ControlTraceStep[];
  facts: ControlTraceFact[];
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
  floatDemandX?: number | null;
  floatDemandY?: number | null;
  maxCountsPerUpdate?: number | null;
  integerCommand: string;
  residual: string;
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
  acceptedCommandCount: number | null;
  lastAcceptedCommand: string;
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
  ready: "输出链路已贯通"
};

const NEXT_ACTION_LABELS: Record<string, string> = {
  start_mainline: "启动主链",
  check_model: "检查模型",
  check_capture_or_model: "检查采集或模型",
  inspect_runtime_ingress: "查看输入状态",
  check_latency: "检查延迟",
  check_targeting: "检查目标选择",
  inspect_control: "查看控制链",
  check_license_or_build: "检查授权或构建",
  enable_output_gate: "打开输出门",
  activate_trigger: "激活触发条件",
  connect_kmnet: "连接 kmNet",
  monitor_output: "观察输出"
};

function outputTraceLabel(trace: RuntimeOutputTraceState): string {
  return OUTPUT_TRACE_LABELS[trace.code] ?? trace.code;
}

function outputTraceNextAction(trace: RuntimeOutputTraceState): string {
  return NEXT_ACTION_LABELS[trace.next_action] ?? (trace.next_action || "继续观察");
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

function buildBatchStep(input: BuildControlTraceInput): ControlTraceStep {
  const freshness = freshState(input.detectionDataAgeMs, input.freshnessThresholdMs);
  const hasRuntimeInput = positive(input.consumedBatches) || positive(input.targetingBatches);
  const hasPublished = positive(input.publishedBatches);
  if (freshness === "stale" && input.runtimeRunning) {
    return {
      id: "batch",
      label: "识别结果",
      state: "blocked",
      value: formatNumber(input.detectionDataAgeMs, 2, "ms"),
      detail: "最新结果超过控制新鲜度阈值，控制链会拒绝过期样本。",
      evidence: `published=${formatInteger(input.publishedBatches)} · consumed=${formatInteger(input.consumedBatches)} · targeting=${formatInteger(input.targetingBatches)}`
    };
  }
  if (hasRuntimeInput) {
    return {
      id: "batch",
      label: "识别结果",
      state: "ready",
      value: `${formatNumber(input.detectionBatchFps, 1)}/s`,
      detail: freshness === "fresh" ? "控制链路已读取新鲜识别结果。" : "控制链路已读取识别结果，等待新鲜样本。",
      evidence: `published=${formatInteger(input.publishedBatches)} · consumed=${formatInteger(input.consumedBatches)} · targeting=${formatInteger(input.targetingBatches)}`
    };
  }
  return {
    id: "batch",
    label: "识别结果",
    state: input.runtimeRunning || hasPublished ? "waiting" : "idle",
    value: formatInteger(input.publishedBatches),
    detail: input.runtimeRunning ? "等待控制链路读取识别结果。" : "主链未运行，暂未产生控制输入。",
    evidence: `age=${formatNumber(input.detectionDataAgeMs, 2, "ms")} · threshold=${formatNumber(input.freshnessThresholdMs, 2, "ms")}`
  };
}

function buildTargetStep(input: BuildControlTraceInput): ControlTraceStep {
  if (input.hasTarget) {
    return {
      id: "target",
      label: "选择主要目标",
      state: "ready",
      value: input.trackId === null ? "已选择" : `track ${Math.trunc(input.trackId)}`,
      detail: input.classLabel ? `${input.classLabel} 进入控制链。` : "已有目标进入控制链。",
      evidence: `候选/合格/选择=${formatInteger(input.rawCandidateCount)}/${formatInteger(input.eligibleCandidateCount)}/${formatInteger(input.selectedTargetCount)} · score=${formatNumber(input.targetScore, 3)}`
    };
  }
  const hasDiagnostics = input.targetPipelineStage || input.targetPipelineCode || input.targetPipelineMessage;
  return {
    id: "target",
    label: "选择主要目标",
    state: hasDiagnostics ? "blocked" : positive(input.targetingBatches) ? "waiting" : "idle",
    value: formatInteger(input.selectedTargetCount),
    detail: input.targetPipelineMessage || "尚未选出可控目标。",
    evidence: input.targetPipelineRejections || input.targetPipelineCode || `detections=${formatInteger(input.detectionCount)}`
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
      label: "跟踪限幅与量化",
      state: input.hasTarget ? "waiting" : "idle",
      value: "—",
      detail: input.hasTarget ? "等待连续控制器产生浮点需求。" : "没有控制需求时限幅器保持空闲。",
      evidence: `limit=${formatNumber(input.maxCountsPerUpdate ?? null, 0, "counts")}`
    };
  }
  const maxCounts = input.maxCountsPerUpdate ?? null;
  const demandX = input.floatDemandX ?? null;
  const demandY = input.floatDemandY ?? null;
  const saturated = maxCounts !== null && (
    (demandX !== null && Math.abs(demandX) > maxCounts)
    || (demandY !== null && Math.abs(demandY) > maxCounts)
  );
  return {
    id: "limiter",
    label: "跟踪限幅与量化",
    state: "ready",
    value: present(input.integerCommand) ? input.integerCommand : "已计算",
    detail: saturated
      ? "本次浮点需求触发单轴上限，随后截断为整数命令并保留量化余量。"
      : "本次浮点需求未触发单轴上限，已截断为整数命令并保留量化余量。",
    evidence: `demand=${input.floatDemand} · limit=${formatNumber(maxCounts, 0, "counts")} · residual=${input.residual}`
  };
}

function buildRecoilStep(input: BuildControlTraceInput): ControlTraceStep {
  if (!input.outputEnabled) {
    return {
      id: "recoil",
      label: "压枪延迟叠加",
      state: "idle",
      value: "未计算",
      detail: "物理输出门关闭，设备边界不会推进压枪延迟或叠加节拍。",
      evidence: "output_enabled=false"
    };
  }
  if (input.triggerActive === false || !input.kmnetRuntimeConnected) {
    return {
      id: "recoil",
      label: "压枪延迟叠加",
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
      label: "压枪延迟叠加",
      state: "ready",
      value: "关闭（旁路）",
      detail: "独立压枪未启用，已准入的跟踪命令在设备边界保持不变。",
      evidence: "recoil_enabled=false"
    };
  }
  if (!input.controllerActive || !input.recoilState) {
    return {
      id: "recoil",
      label: "压枪延迟叠加",
      state: input.hasTarget ? "waiting" : "idle",
      value: "等待样本",
      detail: "等待设备边界报告真实左键、首发延迟和压枪叠加结果。",
      evidence: `remaining=${formatNumber(input.recoilRemainingMs ?? null, 2, "ms")}`
    };
  }
  return {
    id: "recoil",
    label: "压枪延迟叠加",
    state: "ready",
    value: input.recoilStatus || input.recoilState,
    detail: "该阶段只延迟并叠加压枪 +Y，不会延迟左键事件或跟踪控制命令。",
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
      evidence: `full=${input.fullError} · atan=${input.floatDemand} · residual=${input.residual}`
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

function buildGateStep(input: BuildControlTraceInput): ControlTraceStep {
  if (!input.outputEnabled) {
    return {
      id: "gate",
      label: "输出门控",
      state: "idle",
      value: "暂停",
      detail: "物理输出关闭；算法继续计算，但不会发送鼠标偏移。",
      evidence: input.noSendReason || "output_enabled=false"
    };
  }
  if (input.willEmit === true) {
    return {
      id: "gate",
      label: "输出门控",
      state: "ready",
      value: "准入",
      detail: "当前整数命令已通过触发条件和物理输出门，准备进入设备通道。",
      evidence: input.triggerActive === null ? "trigger=未知" : input.triggerActive ? "trigger=按下" : "trigger=未按下"
    };
  }
  const waitingForTrigger = input.noSendReason.includes("触发") || input.triggerActive === false;
  return {
    id: "gate",
    label: "输出门控",
    state: waitingForTrigger || !input.hasTarget ? "waiting" : "blocked",
    value: "未发送",
    detail: input.noSendReason || "控制门控未通过。",
    evidence: input.triggerActive === null ? "trigger=未知" : input.triggerActive ? "trigger=按下" : "trigger=未按下"
  };
}

function buildDeviceStep(input: BuildControlTraceInput): ControlTraceStep {
  if (!input.outputEnabled) {
    return {
      id: "device",
      label: "命令输出",
      state: "idle",
      value: "暂停",
      detail: "输出门关闭时不要求设备回执。",
      evidence: "最近回执不会被解释为当前样本。"
    };
  }
  if (input.kmnetRuntimeConnected && positive(input.acceptedCommandCount)) {
    return {
      id: "device",
      label: "命令输出",
      state: "ready",
      value: input.lastAcceptedCommand,
      detail: "运行时设备通道已接受过控制命令。",
      evidence: `accepted=${formatInteger(input.acceptedCommandCount)} · ${input.kmnetConnectionLabel}`
    };
  }
  if (input.kmnetRuntimeConnected) {
    return {
      id: "device",
      label: "命令输出",
      state: "waiting",
      value: "已连接",
      detail: "设备通道已连接，等待第一条被接受的控制命令。",
      evidence: input.kmnetConnectionLabel
    };
  }
  return {
    id: "device",
    label: "命令输出",
    state: input.willEmit === true ? "blocked" : "waiting",
    value: "未连接",
    detail: input.willEmit === true ? "控制命令已准入，但运行时设备通道未连接。" : "等待输出门准入和 kmNet 运行时连接。",
    evidence: input.deviceLastError || input.kmnetConnectionLabel
  };
}

export function buildControlTrace(input: BuildControlTraceInput): ControlTraceSummary {
  const steps = [
    buildBatchStep(input),
    buildTargetStep(input),
    buildAimStep(input),
    buildControllerStep(input),
    buildLimiterStep(input),
    buildGateStep(input),
    buildRecoilStep(input),
    buildDeviceStep(input)
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
  const detail = input.outputTrace?.detail ||
    "按实时链路串起识别结果、目标速度预测、连续控制、跟踪限幅、输出门控、设备边界压枪叠加和设备回执。";
  const traceFacts: ControlTraceFact[] = input.outputTrace
    ? [
        {
          label: "输出状态",
          value: outputTraceLabel(input.outputTrace),
          detail: outputTraceNextAction(input.outputTrace)
        }
      ]
    : [];

  return {
    state,
    title,
    detail,
    completed,
    total: steps.length,
    steps,
    facts: [
      ...traceFacts,
      {
        label: "当前目标",
        value: input.trackId === null ? (input.hasTarget ? "已选择" : "无") : `track ${Math.trunc(input.trackId)}`,
        detail: input.classLabel || "等待类别"
      },
      {
        label: "控制误差",
        value: formatNumber(input.errorDistancePx, 2, "px"),
        detail: input.controlError
      },
      {
        label: "整数命令",
        value: present(input.integerCommand) ? input.integerCommand : "—",
        detail: input.willEmit === true ? "已准入" : input.noSendReason || "未发送"
      },
      {
        label: "最近回执",
        value: positive(input.acceptedCommandCount) ? input.lastAcceptedCommand : "—",
        detail: `accepted=${formatInteger(input.acceptedCommandCount)}`
      }
    ]
  };
}
