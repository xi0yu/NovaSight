export type ProductConfigState = "live" | "saved" | "restart" | "missing" | "paused";

export type ProductConfigAction = "capture" | "models" | "control" | "kmnet" | "advanced";

export type ProductConfigItemId =
  | "capture"
  | "model"
  | "control"
  | "output"
  | "kmnet"
  | "reload";

export type ProductConfigItem = {
  id: ProductConfigItemId;
  label: string;
  state: ProductConfigState;
  value: string;
  detail: string;
  evidence: string;
  action?: ProductConfigAction;
  actionLabel?: string;
};

export type ProductConfigFact = {
  label: string;
  value: string;
  detail: string;
};

export type ProductConfigProfile = {
  state: ProductConfigState;
  title: string;
  detail: string;
  liveCount: number;
  totalCount: number;
  attentionCount: number;
  items: ProductConfigItem[];
  facts: ProductConfigFact[];
};

export type BuildProductConfigProfileInput = {
  runtimeRunning: boolean;
  configRestartRequired: boolean;
  desiredRevision: number;
  effectiveRevision: number;
  captureConfigured: boolean;
  captureDevice: string;
  captureProfile: string;
  captureSource: string;
  roiLabel: string;
  roiApplied: boolean;
  modelPublished: boolean;
  modelRuntimeLoaded: boolean;
  modelName: string;
  artifactLabel: string;
  backendLabel: string;
  configuredBackendLabel: string;
  modelInputLabel: string;
  postprocessLabel: string;
  postprocessApplied: boolean;
  controlModeLabel: string;
  triggerModeLabel: string;
  predictionEnabled: boolean;
  freshnessThresholdLabel: string;
  outputEnabled: boolean;
  outputRuntimeConnected: boolean;
  kmnetAutoConnect: boolean;
  kmnetRuntimeConnected: boolean;
  kmnetRestartRequired: boolean;
  kmnetConnectionLabel: string;
  kmnetHost: string;
  kmnetPort: string;
  kmnetUuid: string;
};

const EMPTY = "—";

function present(value: string): boolean {
  const normalized = value.trim();
  return normalized.length > 0 && normalized !== EMPTY && normalized !== "-";
}

function revisionLabel(desired: number, effective: number): string {
  if (desired <= 0 && effective <= 0) {
    return "未上报";
  }
  return `${Math.trunc(effective)} / ${Math.trunc(desired)}`;
}

function buildCaptureItem(input: BuildProductConfigProfileInput): ProductConfigItem {
  if (!input.captureConfigured) {
    return {
      id: "capture",
      label: "采集规格",
      state: "missing",
      value: "未选择",
      detail: "缺少明确采集设备或输入格式，启动时只能依赖自动探测。",
      evidence: input.captureDevice || "等待设备",
      action: "capture",
      actionLabel: "选择采集"
    };
  }
  if (input.configRestartRequired && !input.roiApplied) {
    return {
      id: "capture",
      label: "采集规格",
      state: "restart",
      value: input.captureProfile,
      detail: "采集或 ROI 已保存，但运行态仍在使用旧区域。",
      evidence: `${input.captureSource} · ROI ${input.roiLabel}`,
      action: "capture",
      actionLabel: "检查采集"
    };
  }
  if (input.runtimeRunning && input.roiApplied) {
    return {
      id: "capture",
      label: "采集规格",
      state: "live",
      value: input.captureProfile,
      detail: "采集规格与 ROI 已在当前运行状态中生效。",
      evidence: `${input.captureDevice || "设备"} · ROI ${input.roiLabel}`
    };
  }
  return {
    id: "capture",
    label: "采集规格",
    state: "saved",
    value: input.captureProfile,
    detail: "配置已保存，启动主链后会用于采集。",
    evidence: `${input.captureSource} · ROI ${input.roiLabel}`,
    action: "capture",
    actionLabel: "调整采集"
  };
}

function buildModelItem(input: BuildProductConfigProfileInput): ProductConfigItem {
  if (!input.modelPublished || !present(input.artifactLabel)) {
    return {
      id: "model",
      label: "模型与推理",
      state: "missing",
      value: "未发布",
      detail: "需要先发布一个可加载的 TensorRT Engine。",
      evidence: input.modelName,
      action: "models",
      actionLabel: "配置模型"
    };
  }
  if (input.configRestartRequired && !input.postprocessApplied) {
    return {
      id: "model",
      label: "模型与推理",
      state: "restart",
      value: input.modelName,
      detail: "模型或后处理参数已保存，当前运行状态仍在使用旧值。",
      evidence: `${input.artifactLabel} · ${input.postprocessLabel}`,
      action: "models",
      actionLabel: "检查模型"
    };
  }
  if (input.runtimeRunning && input.modelRuntimeLoaded && input.postprocessApplied) {
    return {
      id: "model",
      label: "模型与推理",
      state: "live",
      value: input.modelName,
      detail: "当前运行状态已加载模型，置信度与 NMS 参数一致。",
      evidence: `${input.backendLabel} · ${input.modelInputLabel}`
    };
  }
  return {
    id: "model",
    label: "模型与推理",
    state: "saved",
    value: input.modelName,
    detail: "模型已发布；启动或重载后会进入当前运行状态。",
    evidence: `${input.artifactLabel} · ${input.configuredBackendLabel}`,
    action: "models",
    actionLabel: "模型管理"
  };
}

function buildControlItem(input: BuildProductConfigProfileInput): ProductConfigItem {
  if (input.configRestartRequired) {
    return {
      id: "control",
      label: "控制策略",
      state: "restart",
      value: input.controlModeLabel,
      detail: "控制参数已保存，重启主链后才会完全生效。",
      evidence: `${input.triggerModeLabel} · ${input.predictionEnabled ? "预测开启" : "预测关闭"}`,
      action: "control",
      actionLabel: "检查控制"
    };
  }
  if (input.runtimeRunning) {
    return {
      id: "control",
      label: "控制策略",
      state: "live",
      value: input.controlModeLabel,
      detail: "目标选择、预测与 Atan 控制参数正在生效。",
      evidence: `${input.triggerModeLabel} · 新鲜度 ${input.freshnessThresholdLabel}`
    };
  }
  return {
    id: "control",
    label: "控制策略",
    state: "saved",
    value: input.controlModeLabel,
    detail: "控制策略已保存，启动后会读取识别结果。",
    evidence: `${input.triggerModeLabel} · ${input.predictionEnabled ? "预测开启" : "预测关闭"}`,
    action: "control",
    actionLabel: "调整控制"
  };
}

function buildOutputItem(input: BuildProductConfigProfileInput): ProductConfigItem {
  if (!input.outputEnabled) {
    return {
      id: "output",
      label: "物理输出",
      state: "paused",
      value: "已暂停",
      detail: "算法继续计算，但不会把偏移量交付给设备。",
      evidence: "output_enabled=false",
      action: "control",
      actionLabel: "打开输出"
    };
  }
  if (input.kmnetRestartRequired) {
    return {
      id: "output",
      label: "物理输出",
      state: "restart",
      value: "等待硬件重载",
      detail: "输出门已开启，但 kmNet 新配置需要重启 novasightd。",
      evidence: input.kmnetConnectionLabel,
      action: "kmnet",
      actionLabel: "检查 kmNet"
    };
  }
  if (input.outputRuntimeConnected) {
    return {
      id: "output",
      label: "物理输出",
      state: "live",
      value: "允许发送",
      detail: "最终控制量可以进入设备通道。",
      evidence: input.kmnetConnectionLabel
    };
  }
  return {
    id: "output",
    label: "物理输出",
    state: "saved",
    value: "已允许",
    detail: "配置允许输出，实际发送仍等待主链与 kmNet 连接。",
    evidence: input.kmnetConnectionLabel,
    action: "kmnet",
    actionLabel: "连接设备"
  };
}

function buildKmnetItem(input: BuildProductConfigProfileInput): ProductConfigItem {
  const hasEndpoint = present(input.kmnetHost) && present(input.kmnetUuid) && present(input.kmnetPort);
  if (!hasEndpoint) {
    return {
      id: "kmnet",
      label: "kmNet 设备",
      state: "missing",
      value: "未配对",
      detail: "缺少 host、port 或 uuid，无法建立硬件输出连接。",
      evidence: `${input.kmnetHost || EMPTY}:${input.kmnetPort || EMPTY} · ${input.kmnetUuid || EMPTY}`,
      action: "kmnet",
      actionLabel: "配置设备"
    };
  }
  if (!input.kmnetAutoConnect) {
    return {
      id: "kmnet",
      label: "kmNet 设备",
      state: "paused",
      value: "未启用",
      detail: "自动连接已关闭，主链不会主动接入 kmNet。",
      evidence: `${input.kmnetHost}:${input.kmnetPort}`,
      action: "kmnet",
      actionLabel: "启用连接"
    };
  }
  if (input.kmnetRestartRequired) {
    return {
      id: "kmnet",
      label: "kmNet 设备",
      state: "restart",
      value: "等待重启",
      detail: "设备参数已保存，novasightd 需要重启后装载。",
      evidence: `${input.kmnetHost}:${input.kmnetPort} · ${input.kmnetUuid}`,
      action: "kmnet",
      actionLabel: "测试设备"
    };
  }
  if (input.runtimeRunning && input.kmnetRuntimeConnected) {
    return {
      id: "kmnet",
      label: "kmNet 设备",
      state: "live",
      value: "已连接",
      detail: "设备通道已接入 kmNet。",
      evidence: `${input.kmnetHost}:${input.kmnetPort}`
    };
  }
  return {
    id: "kmnet",
    label: "kmNet 设备",
    state: "saved",
    value: input.kmnetConnectionLabel,
    detail: "设备配置已保存，等待主链运行或手动连接验证。",
    evidence: `${input.kmnetHost}:${input.kmnetPort} · ${input.kmnetUuid}`,
    action: "kmnet",
    actionLabel: "测试设备"
  };
}

function buildReloadItem(input: BuildProductConfigProfileInput): ProductConfigItem {
  if (input.configRestartRequired) {
    return {
      id: "reload",
      label: "重载边界",
      state: "restart",
      value: "等待重启",
      detail: "已保存配置版本高于运行态有效版本。",
      evidence: `effective/desired=${revisionLabel(input.desiredRevision, input.effectiveRevision)}`,
      action: "advanced",
      actionLabel: "查看高级"
    };
  }
  return {
    id: "reload",
    label: "重载边界",
    state: input.runtimeRunning ? "live" : "saved",
    value: "版本一致",
    detail: input.runtimeRunning ? "运行态与保存配置版本一致。" : "保存配置已就绪，等待主链启动。",
    evidence: `effective/desired=${revisionLabel(input.desiredRevision, input.effectiveRevision)}`
  };
}

function profileTitle(state: ProductConfigState): string {
  if (state === "missing") return "配置缺口需要处理";
  if (state === "restart") return "配置已保存，等待运行态重载";
  if (state === "paused") return "配置可运行，但输出处于暂停";
  if (state === "live") return "配置正在生效";
  return "配置已保存";
}

function profileDetail(state: ProductConfigState, attentionCount: number): string {
  if (state === "missing") {
    return "先处理缺失项，再启动主链；界面会把用户带到对应配置位置。";
  }
  if (state === "restart") {
    return "当前有已保存但未进入运行态的配置，重启主链后才能验证效果。";
  }
  if (state === "paused") {
    return "采集、模型和控制可以继续验证，物理输出保持人工暂停。";
  }
  if (state === "live") {
    return "采集、模型、控制、设备和配置版本形成了同一条有效链路。";
  }
  return attentionCount > 0
    ? "配置已经落盘，部分项目需要启动主链后才能转为生效。"
    : "这些配置会在下一次主链启动时作为默认运行参数使用。";
}

function deriveProfileState(items: ProductConfigItem[]): ProductConfigState {
  if (items.some((item) => item.state === "missing")) return "missing";
  if (items.some((item) => item.state === "restart")) return "restart";
  if (items.some((item) => item.state === "paused")) return "paused";
  if (items.every((item) => item.state === "live")) return "live";
  return "saved";
}

export function buildProductConfigProfile(input: BuildProductConfigProfileInput): ProductConfigProfile {
  const items = [
    buildCaptureItem(input),
    buildModelItem(input),
    buildControlItem(input),
    buildOutputItem(input),
    buildKmnetItem(input),
    buildReloadItem(input)
  ];
  const state = deriveProfileState(items);
  const liveCount = items.filter((item) => item.state === "live").length;
  const attentionCount = items.filter((item) =>
    item.state === "missing" || item.state === "restart"
  ).length;
  const facts: ProductConfigFact[] = [
    {
      label: "配置版本",
      value: revisionLabel(input.desiredRevision, input.effectiveRevision),
      detail: input.configRestartRequired ? "运行态仍在旧版本" : "保存值与运行态一致"
    },
    {
      label: "推理方式",
      value: input.backendLabel,
      detail: input.configuredBackendLabel === input.backendLabel ? "与配置一致" : `配置值 ${input.configuredBackendLabel}`
    },
    {
      label: "当前模型",
      value: input.modelName,
      detail: input.artifactLabel || "等待发布"
    },
    {
      label: "输出状态",
      value: input.outputEnabled ? "允许" : "暂停",
      detail: input.kmnetConnectionLabel
    }
  ];
  return {
    state,
    title: profileTitle(state),
    detail: profileDetail(state, attentionCount),
    liveCount,
    totalCount: items.length,
    attentionCount,
    items,
    facts
  };
}
