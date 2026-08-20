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
  const telemetryWindowMs = statistics?.telemetry_window_ms ?? null;
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
  const failed =
    fatalMessage !== ""
      ? true
      : terminalError ||
        pipelineLastError !== "";
  const pipelineState = pipeline?.state ?? "";
  const normalizedFailure = failureMessage.toLowerCase();
  const selectedExecutor = runtime?.executor.selected ?? "";
  const selectedDevice = runtime?.executor.executors[selectedExecutor]
    ?? runtime?.executor.executors.kmnet;
  const visionControl = runtime?.vision.control;
  const outputTrace = runtime?.vision.output_trace ?? null;
  const controlDeviceDisconnected =
    running &&
    selectedExecutor !== "replay" &&
    visionControl?.output_enabled === true &&
    selectedDevice?.runtime_connected !== true;

  let readinessCode: RuntimeMainlineStatus["readinessCode"] = "stopped";
  let readinessLabel = "未启动";
  let readinessDetail = "点击启动后，系统会自动检查采集、模型、推理和控制链路。";
  if (failed) {
    if (
      normalizedFailure.includes("no admitted batch") &&
      !positive(nvinferInputFrames) &&
      !positive(metadataExtractions)
    ) {
      readinessCode = "no_video";
      readinessLabel = "未检测到画面";
      readinessDetail = "采集链在启动时没有收到有效画面；请检查采集卡信号、输入格式和分辨率。";
    } else if (
      normalizedFailure.includes("model") ||
      normalizedFailure.includes("engine") ||
      normalizedFailure.includes("tensorrt") ||
      normalizedFailure.includes("nvinfer")
    ) {
      readinessCode = "model_load_failed";
      readinessLabel = "模型加载失败";
      readinessDetail = "请检查当前 Engine 是否由本机 Jetson 生成，并确认模型输入、输出和解析器配置匹配。";
    } else if (
      normalizedFailure.includes("freshness") ||
      normalizedFailure.includes("frame age") ||
      normalizedFailure.includes("latency") ||
      normalizedFailure.includes("stale")
    ) {
      readinessCode = "frame_latency_high";
      readinessLabel = "当前画面延迟过高";
      readinessDetail = "系统已为安全起见拒绝过期结果；请检查采集时间戳、积压和新鲜度阈值。";
    } else {
      readinessCode = "failed";
      readinessLabel = "启动失败";
      readinessDetail = failureMessage || "NovaSight 服务未能完成主链启动，请查看异常信息中的处理建议。";
    }
  } else if (pipelineState === "starting") {
    readinessCode = "starting";
    readinessLabel = "正在启动";
    readinessDetail = "正在加载采集、模型推理和控制链路。";
  } else if (runtime?.semantic?.phase === "waiting_model") {
    readinessCode = "starting";
    readinessLabel = "主链运行 · 等待模型";
    readinessDetail = "控制主链已经启动；发布可用 Engine 后，感知链会在当前进程自动接入。";
  } else if (running && !hasInferenceSignal) {
    readinessCode = "no_video";
    readinessLabel = "未检测到画面";
    readinessDetail = "主链已运行，但尚未收到有效采集帧；请检查采集卡信号和输入配置。";
  } else if (running && telemetryWindowMs !== null && nvinferInputFps === 0) {
    readinessCode = "no_video";
    readinessLabel = "画面输入已停止";
    readinessDetail = "历史上收到过采集帧，但当前统计窗口内没有新的推理输入。";
  } else if (
    running &&
    detectionDataAgeMs !== null &&
    detectionFreshnessThresholdMs !== null &&
    detectionDataAgeMs > detectionFreshnessThresholdMs
  ) {
    readinessCode = "frame_latency_high";
    readinessLabel = "当前画面延迟过高";
    readinessDetail = `最新结果帧龄 ${detectionDataAgeMs.toFixed(2)} ms，已超过控制安全阈值 ${detectionFreshnessThresholdMs.toFixed(2)} ms。`;
  } else if (controlDeviceDisconnected) {
    readinessCode = "control_device_disconnected";
    readinessLabel = "视觉已就绪 · 控制设备未连接";
    readinessDetail = "采集与推理继续运行，鼠标输出保持安全禁用；控制设备恢复后会自动重连。";
  } else if (running && hasRuntimeConsumption && hasFreshDetectionData) {
    readinessCode = "ready";
    readinessLabel = "已就绪";
    readinessDetail =
      outputTrace?.code === "ready" && outputTrace.detail
        ? outputTrace.detail
        : "采集、推理和运行时消费链路正在产生新鲜的真实数据。";
  } else if (running) {
    readinessCode = "starting";
    readinessLabel = "等待运行数据";
    readinessDetail =
      outputTrace?.detail ||
      (positive(publishedBatches)
        ? "已收到识别结果，正在确认数据新鲜度。"
        : "推理链已启动，正在等待首个识别结果。");
  }

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
