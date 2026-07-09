import type { RuntimeState } from "../../api";

export type RuntimeMainlineStatus = {
  running: boolean;
  failed: boolean;
  failureMessage: string;
  pipelineLastError: string;
  terminalError: boolean;
  publishedBatches: number;
  staleDroppedBatches: number;
  windowStaleDroppedBatches: number;
  maxPublishAgeMs: number;
  lastPtsToProbeMs: number;
  tensorMetaFrames: number;
  postprocessFrames: number;
  inferenceCounter: number;
  hasInferenceSignal: boolean;
  hasRuntimeConsumption: boolean;
  progressSummary: string;
};

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function readString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function readBoolean(value: unknown): boolean {
  return typeof value === "boolean" ? value : false;
}

function readNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function maxNumber(...values: unknown[]): number {
  return Math.max(0, ...values.map(readNumber));
}

export function getRuntimeMainlineStatus(runtime: RuntimeState | null): RuntimeMainlineStatus {
  const pipeline = asRecord(runtime?.pipeline);
  const inference = asRecord(runtime?.inference);
  const capture = asRecord(runtime?.capture);
  const statistics = asRecord(runtime?.statistics);
  const captureStatistics = asRecord(capture.statistics);
  const fatal = asRecord(runtime?.fatal_error);
  const pipelineLastError = readString(pipeline.last_error);
  const terminalError = readBoolean(inference.terminal_error);
  const failureMessage =
    pipelineLastError ||
    readString(inference.detail) ||
    readString(inference.reason) ||
    readString(fatal.message);
  const publishedBatches = 0;
  const staleDroppedBatches = 0;
  const windowStaleDroppedBatches = 0;
  const maxPublishAgeMs = 0;
  const lastPtsToProbeMs = 0;
  const tensorMetaFrames = 0;
  const postprocessFrames = 0;
  const inferenceCounter = maxNumber(
    statistics.inference_counter,
    captureStatistics.inference_counter,
    pipeline.processed_frames
  );
  const hasInferenceSignal =
    inferenceCounter > 0 ||
    maxNumber(statistics.inference_fps, captureStatistics.inference_fps, pipeline.inference_fps) > 0;
  const hasRuntimeConsumption =
    inferenceCounter > 0 ||
    maxNumber(statistics.control_observation_counter, captureStatistics.control_observation_counter) > 0 ||
    maxNumber(statistics.inference_fps, captureStatistics.inference_fps, pipeline.inference_fps) > 0;
  const resolvedFailureMessage = failureMessage;
  const progressSummary = [
    `latest=${maxNumber(pipeline.consumed_frames, statistics.capture_counter, captureStatistics.capture_counter)}`,
    `inferred=${inferenceCounter}`,
    `consumed=${inferenceCounter}`
  ].join(" · ");
  const running = runtime?.running === true || pipeline.running === true;
  const failed =
    runtime?.fatal_error !== null && runtime?.fatal_error !== undefined
      ? true
      : terminalError ||
        pipelineLastError !== "" ||
        failureMessage !== "";

  return {
    running,
    failed,
    failureMessage: resolvedFailureMessage,
    pipelineLastError,
    terminalError,
    publishedBatches,
    staleDroppedBatches,
    windowStaleDroppedBatches,
    maxPublishAgeMs,
    lastPtsToProbeMs,
    tensorMetaFrames,
    postprocessFrames,
    inferenceCounter,
    hasInferenceSignal,
    hasRuntimeConsumption,
    progressSummary
  };
}

export function isRuntimeMainlineRunning(runtime: RuntimeState | null): boolean {
  return getRuntimeMainlineStatus(runtime).running;
}
