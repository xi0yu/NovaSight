import type { RuntimeState } from "../../api";
import { getRuntimeMainlineStatus } from "../shared/runtimeStatus";

export type RuntimeTransportConfidence = "current" | "stale" | "unavailable";
export type LifecycleProjectionState = RuntimeState["semantic"]["phase"];
export type PerceptionProjectionState = "unavailable" | "stopped" | "starting" | "waiting_model" | "current" | "stale" | "faulted";
export type OutputProjectionState = "safe" | "blocked" | "armed" | "unknown";

export type RuntimeRecoveryAction =
  | "open-license"
  | "open-model-manager"
  | "open-capture"
  | "open-control"
  | "open-latency"
  | "open-kmnet-test"
  | "open-params";

export type RuntimeProjection = {
  transport: RuntimeTransportConfidence;
  lifecycle: { state: LifecycleProjectionState; label: string; detail: string };
  perception: { state: PerceptionProjectionState; label: string; detail: string };
  output: { state: OutputProjectionState; label: string; detail: string };
  conclusion: string;
  daemonConfirmedSafe: boolean;
  nextAction?: RuntimeRecoveryAction;
  nextActionLabel?: string;
};

const LIFECYCLE_LABELS: Record<LifecycleProjectionState, string> = {
  stopped: "已停止",
  starting: "正在启动",
  waiting_model: "运行中 · 等待模型",
  running: "运行中",
  standby: "待机",
  stopping: "正在停止",
  faulted: "运行故障",
};

export function isDaemonConfirmedSafe(runtime: RuntimeState | null): boolean {
  if (!runtime) return false;
  return runtime.semantic.phase === "stopped"
    && runtime.vision.output_trace.code === "runtime_stopped"
    && runtime.executor.executors.kmnet?.runtime_connected === false
    && runtime.vision.control.will_emit !== true;
}

function perceptionProjection(
  runtime: RuntimeState,
  transport: RuntimeTransportConfidence,
): RuntimeProjection["perception"] {
  if (transport !== "current") {
    return {
      state: transport === "stale" ? "stale" : "unavailable",
      label: transport === "stale" ? "状态已过期" : "状态不可用",
      detail: "无法确认当前感知数据，请先恢复与 novasightd 的实时状态连接。",
    };
  }

  const phase = runtime.semantic.perception_phase;
  if (phase === "faulted") {
    return { state: "faulted", label: "感知故障", detail: runtime.inference.detail || runtime.inference.reason || "推理链发生故障。" };
  }
  if (phase === "waiting_model") {
    return { state: "waiting_model", label: "等待模型", detail: "主链已运行，正在等待可用的模型 Engine。" };
  }
  if (phase === "starting") {
    return { state: "starting", label: "正在建立感知", detail: "采集和推理链正在启动。" };
  }
  if (phase === "unavailable") {
    return { state: "unavailable", label: "感知不可用", detail: "当前配置没有可用的感知链。" };
  }
  if (phase === "stopped") {
    return { state: "stopped", label: "感知已停止", detail: "主链停止时不会产生新的感知数据。" };
  }

  const age = runtime.statistics.detection_data_age_ms;
  const threshold = runtime.statistics.detection_freshness_threshold_ms;
  if (age === null || threshold === null || age > threshold) {
    return {
      state: "stale",
      label: "感知数据过期",
      detail: age === null || threshold === null
        ? "尚未获得可验证的新鲜度数据。"
        : `最新结果帧龄 ${age.toFixed(1)} ms，超过安全阈值 ${threshold.toFixed(1)} ms。`,
    };
  }
  return { state: "current", label: "感知数据新鲜", detail: `最新结果帧龄 ${age.toFixed(1)} ms。` };
}

function recoveryAction(runtime: RuntimeState): Pick<RuntimeProjection, "nextAction" | "nextActionLabel"> {
  if (getRuntimeMainlineStatus(runtime).readinessCode === "no_video") {
    return { nextAction: "open-capture", nextActionLabel: "检查采集" };
  }
  switch (runtime.vision.output_trace.next_action) {
    case "check_model":
      return { nextAction: "open-model-manager", nextActionLabel: "检查模型" };
    case "check_capture_or_model":
      return runtime.active_model
        ? { nextAction: "open-capture", nextActionLabel: "检查采集" }
        : { nextAction: "open-model-manager", nextActionLabel: "配置模型" };
    case "check_latency":
      return { nextAction: "open-latency", nextActionLabel: "检查延迟" };
    case "enable_output_gate":
      return { nextAction: "open-params", nextActionLabel: "配置输出" };
    case "connect_kmnet":
      return { nextAction: "open-kmnet-test", nextActionLabel: "检查 kmNet" };
    case "check_license_or_build":
      return { nextAction: "open-license", nextActionLabel: "检查授权" };
    case "check_targeting":
    case "inspect_control":
    case "inspect_runtime_ingress":
    case "activate_trigger":
      return { nextAction: "open-control", nextActionLabel: "查看控制" };
    default:
      return {};
  }
}

export function projectRuntimeState(
  runtime: RuntimeState | null,
  transport: RuntimeTransportConfidence,
): RuntimeProjection | null {
  if (!runtime) return null;
  const daemonConfirmedSafe = isDaemonConfirmedSafe(runtime);
  const perception = perceptionProjection(runtime, transport);
  const lifecycleUncertain = ["starting", "stopping", "faulted"].includes(runtime.semantic.phase)
    || (runtime.semantic.phase === "stopped" && !daemonConfirmedSafe);
  const output: RuntimeProjection["output"] = transport !== "current"
    ? { state: "unknown", label: "输出状态未知", detail: "实时状态不可用，不能推断硬件输出是否安全。" }
    : lifecycleUncertain
      ? { state: "unknown", label: "输出状态未知", detail: "生命周期异常或停止证据不完整，不能推断硬件输出状态。" }
      : perception.state === "stale" || perception.state === "faulted" || perception.state === "unavailable"
        ? { state: "blocked", label: "输出已阻止", detail: "感知状态不可用或已过期，输出必须保持阻止。" }
        : daemonConfirmedSafe
          ? { state: "safe", label: "已确认安全", detail: "运行态已停止、kmNet 已断开，且不存在待发送输出。" }
          : runtime.vision.control.will_emit === true || runtime.vision.output_trace.code === "ready"
            ? { state: "armed", label: "输出已具备条件", detail: runtime.vision.output_trace.detail || "当前样本满足输出门控条件。" }
            : { state: "blocked", label: "输出已阻止", detail: runtime.vision.output_trace.detail || "当前输出门控未满足。" };
  const lifecycle = {
    state: runtime.semantic.phase,
    label: LIFECYCLE_LABELS[runtime.semantic.phase],
    detail: runtime.fatal_error?.message || `novasightd 快照 #${runtime.semantic.snapshot_sequence}`,
  };
  const conclusion = transport !== "current"
    ? "无法确认实时运行状态"
    : runtime.semantic.phase === "faulted"
      ? "运行链发生故障，需要处理后再恢复运行"
      : daemonConfirmedSafe
        ? "系统已停止并确认硬件输出安全"
        : output.state === "armed"
          ? "主链正在运行，输出条件已满足"
          : `${lifecycle.label}，${perception.label}，${output.label}`;
  return {
    transport,
    lifecycle,
    perception,
    output,
    conclusion,
    daemonConfirmedSafe,
    ...recoveryAction(runtime),
  };
}
