import type { LicenseStatus, RuntimeState } from "../../api";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";

export type LaunchReadinessState = "ready" | "action" | "blocked" | "idle";

export type LaunchReadinessAction =
  | "open-license"
  | "open-model-manager"
  | "open-capture"
  | "open-control"
  | "open-latency"
  | "open-kmnet-test"
  | "open-params";

export type LaunchReadinessSummary = {
  state: LaunchReadinessState;
  title: string;
  detail: string;
  primaryAction?: LaunchReadinessAction;
  primaryActionLabel?: string;
};

function actionForRuntime(runtime: RuntimeState): Pick<
  LaunchReadinessSummary,
  "primaryAction" | "primaryActionLabel"
> {
  const nextAction = runtime.vision.output_trace?.next_action;
  switch (nextAction) {
    case "check_model":
      return { primaryAction: "open-model-manager", primaryActionLabel: "检查模型" };
    case "check_capture_or_model":
      return runtime.active_model
        ? { primaryAction: "open-capture", primaryActionLabel: "检查采集" }
        : { primaryAction: "open-model-manager", primaryActionLabel: "配置模型" };
    case "check_latency":
      return { primaryAction: "open-latency", primaryActionLabel: "检查延迟" };
    case "enable_output_gate":
      return { primaryAction: "open-params", primaryActionLabel: "配置输出" };
    case "connect_kmnet":
      return { primaryAction: "open-kmnet-test", primaryActionLabel: "配置 kmNet" };
    case "check_license_or_build":
      return { primaryAction: "open-license", primaryActionLabel: "检查授权" };
    case "check_targeting":
    case "inspect_control":
    case "inspect_runtime_ingress":
    case "activate_trigger":
      return { primaryAction: "open-control", primaryActionLabel: "查看控制" };
    default:
      return {};
  }
}

export function buildLaunchReadiness({
  license,
  runtime,
  serviceAvailable
}: {
  license: LicenseStatus | null;
  runtime: RuntimeState | null;
  serviceAvailable: boolean | null;
}): LaunchReadinessSummary {
  if (serviceAvailable !== true) {
    return {
      state: serviceAvailable === false ? "blocked" : "idle",
      title: serviceAvailable === false ? "服务不可达" : "正在连接服务",
      detail: serviceAvailable === false
        ? "Studio 无法确认 NovaSight 当前运行状态，运行操作已暂停。"
        : "Studio 正在等待 NovaSight 返回连接状态。"
    };
  }
  if (!license?.valid) {
    return {
      state: "blocked",
      title: "等待授权",
      detail: license?.message || "激活授权后即可使用已保存配置运行主链。",
      primaryAction: "open-license",
      primaryActionLabel: "处理授权"
    };
  }
  if (!runtime) {
    return {
      state: "idle",
      title: "正在读取运行状态",
      detail: "Studio 正在等待 NovaSight 返回权威运行状态。"
    };
  }

  const status = getRuntimeMainlineStatus(runtime);
  const phase = runtime.semantic.phase;
  if (phase === "starting") {
    return {
      state: "idle",
      title: "正在启动",
      detail: "NovaSight 正在使用已保存配置启动采集、推理和控制链。"
    };
  }
  if (phase === "stopping") {
    return {
      state: "idle",
      title: "正在停止",
      detail: "NovaSight 正在关闭运行链并失效待发送输出。"
    };
  }
  if (phase === "faulted" || status.failed) {
    return {
      state: "blocked",
      title: status.readinessLabel || "运行失败",
      detail: status.readinessDetail || runtime.fatal_error?.message || "运行链发生错误。",
      ...actionForRuntime(runtime)
    };
  }
  if (phase === "waiting_model") {
    return {
      state: "action",
      title: "主链运行 · 等待模型",
      detail: "采集与控制运行态已建立；发布可用 TensorRT Engine 后会自动接入。",
      primaryAction: "open-model-manager",
      primaryActionLabel: "配置模型"
    };
  }
  if (phase === "stopped") {
    return {
      state: "ready",
      title: "可以运行",
      detail: runtime.active_model
        ? "点击运行后，NovaSight 会直接使用已保存配置启动主链。"
        : "点击运行后主链会先启动并等待模型，不需要填写启动参数。"
    };
  }

  const action = actionForRuntime(runtime);
  if (action.primaryAction) {
    return {
      state: runtime.vision.output_trace?.state === "blocked" ? "action" : "ready",
      title: status.readinessLabel || "主链运行中",
      detail: status.readinessDetail || runtime.vision.output_trace?.detail || "运行状态正常。",
      ...action
    };
  }
  return {
    state: "ready",
    title: "主链运行中",
    detail: status.readinessDetail || "采集、推理和控制链正在运行。"
  };
}
