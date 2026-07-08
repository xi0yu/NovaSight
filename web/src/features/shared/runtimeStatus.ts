import type { RuntimeState } from "../../api";

export type RuntimeMainlineStatus = {
  running: boolean;
  failed: boolean;
  failureMessage: string;
  pipelineLastError: string;
  terminalError: boolean;
  publishedBatches: number;
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
  const detectionSource = asRecord(pipeline.detection_source);
  const inference = asRecord(runtime?.inference);
  const capture = asRecord(runtime?.capture);
  const statistics = asRecord(runtime?.statistics);
  const captureStatistics = asRecord(capture.statistics);
  const fatal = asRecord(runtime?.fatal_error);
  const pipelineLastError = readString(pipeline.last_error);
  const terminalError = readBoolean(inference.terminal_error);
  const failureMessage =
    pipelineLastError ||
    readString(detectionSource.last_error) ||
    readString(detectionSource.detail) ||
    readString(detectionSource.reason) ||
    readString(inference.detail) ||
    readString(inference.reason) ||
    readString(fatal.message);
  const publishedBatches = maxNumber(
    statistics.published_batches,
    captureStatistics.published_batches,
    detectionSource.published_batches
  );
  const tensorMetaFrames = maxNumber(
    statistics.tensor_meta_frames,
    captureStatistics.tensor_meta_frames,
    detectionSource.tensor_meta_frames
  );
  const postprocessFrames = maxNumber(
    statistics.postprocess_frames,
    captureStatistics.postprocess_frames,
    detectionSource.postprocess_frames
  );
  const inferenceCounter = maxNumber(
    statistics.inference_counter,
    captureStatistics.inference_counter,
    pipeline.processed_frames,
    pipeline.consumed_detection_batches
  );
  const hasInferenceSignal =
    publishedBatches > 0 ||
    tensorMetaFrames > 0 ||
    postprocessFrames > 0 ||
    maxNumber(statistics.detection_batch_fps, captureStatistics.detection_batch_fps, detectionSource.detection_batch_fps) > 0 ||
    maxNumber(statistics.tensor_meta_fps, captureStatistics.tensor_meta_fps, detectionSource.tensor_meta_fps) > 0 ||
    maxNumber(statistics.postprocess_fps, captureStatistics.postprocess_fps, detectionSource.postprocess_fps) > 0;
  const hasRuntimeConsumption =
    inferenceCounter > 0 ||
    maxNumber(statistics.control_observation_counter, captureStatistics.control_observation_counter) > 0 ||
    maxNumber(statistics.inference_fps, captureStatistics.inference_fps, pipeline.inference_fps) > 0;
  const progressSummary = [
    `published=${publishedBatches}`,
    `tensor=${tensorMetaFrames}`,
    `postprocess=${postprocessFrames}`,
    `consumed=${inferenceCounter}`
  ].join(" · ");
  const running = runtime?.running === true || pipeline.running === true;
  const failed =
    runtime?.fatal_error !== null && runtime?.fatal_error !== undefined
      ? true
      : terminalError ||
        pipelineLastError !== "" ||
        (detectionSource.available === false && (running || failureMessage !== "")) ||
        (detectionSource.running === false && running && failureMessage !== "");

  return {
    running,
    failed,
    failureMessage,
    pipelineLastError,
    terminalError,
    publishedBatches,
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
