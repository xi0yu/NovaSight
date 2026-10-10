import type { RuntimeOutputTraceState, RuntimeState } from "../../api";

export type RuntimeMainlineStatus = {
  running: boolean;
  failed: boolean;
  failureMessage: string;
  pipelineLastError: string;
  terminalError: boolean;
  nvinferInputFrames: number | null;
  metadataExtractions: number | null;
  publishedBatches: number | null;
  consumedBatches: number | null;
  targetingBatches: number | null;
  nvinferInputFps: number | null;
  detectionBatchFps: number | null;
  detectionDataAgeMs: number | null;
  detectionFreshnessThresholdMs: number | null;
  hasInferenceSignal: boolean;
  hasRuntimeConsumption: boolean;
  hasFreshDetectionData: boolean;
  outputTrace: RuntimeOutputTraceState | null;
  progressSummary: string;
  readinessCode:
    | "stopped"
    | "starting"
    | "waiting_model"
    | "ready"
    | "no_video"
    | "control_device_disconnected"
    | "model_load_failed"
    | "frame_latency_high"
    | "failed";
  readinessLabel: string;
  readinessDetail: string;
};

function firstNumber(...values: Array<number | null | undefined>): number | null {
  for (const value of values) {
    if (value !== null && value !== undefined) {
      return value;
    }
  }
  return null;
}

function positive(value: number | null): boolean {
  return value !== null && value > 0;
}

function counterLabel(value: number | null): string {
  return value === null ? "—" : String(Math.trunc(value));
}

export function getRuntimeMainlineStatus(runtime: RuntimeState | null): RuntimeMainlineStatus {
  const pipeline = runtime?.pipeline;
  const deepstream = pipeline?.deepstream;
  const inference = runtime?.inference;
  const statistics = runtime?.statistics;
  const terminalError = inference?.terminal_error === true || deepstream?.terminal_error === true;
  const pipelineLastError =
    pipeline?.last_error?.message ?? (terminalError ? deepstream?.last_error ?? "" : "");
  const fatalMessage = runtime?.fatal_error?.message ?? "";
  const failureMessage =
    pipelineLastError ||
    fatalMessage ||
    (terminalError ? inference?.detail || inference?.reason || "" : "");
  const nvinferInputFrames = firstNumber(
    statistics?.nvinfer_input_counter,
    deepstream?.input_frames,
    inference?.input_frames
  );
  const metadataExtractions = firstNumber(
    deepstream?.metadata_extractions,
    inference?.metadata_extractions
  );
  const publishedBatches = firstNumber(
    statistics?.detection_batch_counter,
    deepstream?.published_batches,
    inference?.published_batches
  );
  const consumedBatches = firstNumber(
    statistics?.detection_batch_consumed_counter
  );
  const targetingBatches = firstNumber(
    statistics?.targeting_batch_counter
  );
  const detectionBatchFps = statistics?.detection_batch_fps ?? null;
  const nvinferInputFps = statistics?.nvinfer_input_fps ?? null;
  const detectionDataAgeMs = statistics?.detection_data_age_ms ?? null;
  const detectionFreshnessThresholdMs = statistics?.detection_freshness_threshold_ms ?? null;
  const hasInferenceSignal =
    positive(nvinferInputFrames) ||
    positive(metadataExtractions) ||
    positive(publishedBatches);
  const hasRuntimeConsumption =
    positive(consumedBatches) || positive(targetingBatches);
  const hasFreshDetectionData =
    detectionDataAgeMs !== null &&
    detectionFreshnessThresholdMs !== null &&
    detectionDataAgeMs <= detectionFreshnessThresholdMs;
  const resolvedFailureMessage = failureMessage;
  const progressSummary = [
    `推理输入=${counterLabel(nvinferInputFrames)}`,
    `元数据=${counterLabel(metadataExtractions)}`,
    `识别结果=${counterLabel(publishedBatches)}`,
    `控制读取=${counterLabel(consumedBatches)}`,
    `目标选择=${counterLabel(targetingBatches)}`
  ].join(" · ");
  const running = runtime?.running === true || pipeline?.running === true || deepstream?.running === true;
  const outputTrace = runtime?.vision.output_trace ?? null;
  const readinessCode = runtime?.presentation.readiness.code ?? "stopped";
  const failed = readinessCode === "failed" || readinessCode === "model_load_failed";
  const [readinessLabel, readinessDetail] = (() => {
    switch (readinessCode) {
      case "starting":
        return ["正在启动", "正在加载采集、模型推理和控制链路。"];
      case "waiting_model":
        return ["主链运行 · 等待模型", "发布可用 Engine 后，感知链会自动接入。"];
      case "ready":
        return ["已就绪", outputTrace?.detail || "采集、推理和控制链正在处理新鲜数据。"];
      case "no_video":
        return ["未检测到画面", "请检查采集卡信号、输入格式和分辨率。"];
      case "control_device_disconnected":
        return ["视觉已就绪 · 控制设备未连接", "鼠标输出保持安全禁用；请检查控制设备连接。"];
      case "model_load_failed":
        return ["模型加载失败", failureMessage || "请检查 Engine 与当前 Jetson、模型输入输出和解析器配置。"];
      case "frame_latency_high": return ["当前画面延迟过高", detectionDataAgeMs === null || detectionFreshnessThresholdMs === null
        ? "系统已拒绝过期结果，请检查采集时间戳和数据积压。"
        : `最新结果帧龄 ${detectionDataAgeMs.toFixed(2)} ms，超过安全阈值 ${detectionFreshnessThresholdMs.toFixed(2)} ms。`];
      case "failed":
        return ["运行失败", failureMessage || "NovaSight 运行链发生故障，请查看异常信息。"];
      default:
        return ["未启动", "打开首页运行总开关后，系统会使用已保存配置开始工作。"];
    }
  })();

  return {
    running,
    failed,
    failureMessage: resolvedFailureMessage,
    pipelineLastError,
    terminalError,
    nvinferInputFrames,
    metadataExtractions,
    publishedBatches,
    consumedBatches,
    targetingBatches,
    nvinferInputFps,
    detectionBatchFps,
    detectionDataAgeMs,
    detectionFreshnessThresholdMs,
    hasInferenceSignal,
    hasRuntimeConsumption,
    hasFreshDetectionData,
    outputTrace,
    progressSummary,
    readinessCode,
    readinessLabel,
    readinessDetail
  };
}

export function isRuntimeMainlineRunning(runtime: RuntimeState | null): boolean {
  return getRuntimeMainlineStatus(runtime).running;
}
