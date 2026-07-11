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
  consumedBatches: number;
  controlObservations: number;
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
  const deepstream = asRecord(pipeline.deepstream);
  const inference = asRecord(runtime?.inference);
  const capture = asRecord(runtime?.capture);
  const statistics = asRecord(runtime?.statistics);
  const captureStatistics = asRecord(capture.statistics);
  const fatal = asRecord(runtime?.fatal_error);
  const terminalError = readBoolean(inference.terminal_error) || readBoolean(deepstream.terminal_error);
  const pipelineLastError =
    readString(pipeline.last_error) || (terminalError ? readString(deepstream.last_error) : "");
  const fatalMessage = readString(fatal.message);
  const failureMessage =
    pipelineLastError ||
    fatalMessage ||
    (terminalError ? readString(inference.detail) || readString(inference.reason) : "");
  const publishedBatches = maxNumber(
    deepstream.published_batches,
    asRecord(deepstream.detection_batch_mailbox).published_batches
  );
  const staleDroppedBatches = readNumber(deepstream.stale_dropped_batches);
  const windowStaleDroppedBatches = 0;
  const maxPublishAgeMs = readNumber(deepstream.max_publish_age_ms);
  const lastPtsToProbeMs = 0;
  const tensorMetaFrames = 0;
  const postprocessFrames = readNumber(deepstream.object_meta_frames);
  const consumedBatches = maxNumber(
    pipeline.consumed_detection_batches,
    pipeline.processed_frames,
    statistics.inference_counter
  );
  const controlObservations = maxNumber(
    pipeline.control_observations,
    statistics.control_observation_counter,
    captureStatistics.control_observation_counter
  );
  const inferenceCounter = maxNumber(
    statistics.inference_counter,
    captureStatistics.inference_counter,
    pipeline.processed_frames,
    pipeline.consumed_detection_batches,
    deepstream.object_meta_frames
  );
  const hasInferenceSignal =
    inferenceCounter > 0 ||
    publishedBatches > 0 ||
    maxNumber(statistics.inference_fps, captureStatistics.inference_fps, pipeline.inference_fps) > 0;
  const hasRuntimeConsumption =
    consumedBatches > 0 || controlObservations > 0;
  const resolvedFailureMessage = failureMessage;
  const progressSummary = [
    `input=${maxNumber(deepstream.input_frames, statistics.capture_counter, captureStatistics.capture_counter)}`,
    `published=${publishedBatches}`,
    `consumed=${consumedBatches}`,
    `control=${controlObservations}`
  ].join(" · ");
  const running = runtime?.running === true || pipeline.running === true || deepstream.running === true;
  const failed =
    fatalMessage !== ""
      ? true
      : terminalError ||
        pipelineLastError !== "";

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
    consumedBatches,
    controlObservations,
    inferenceCounter,
    hasInferenceSignal,
    hasRuntimeConsumption,
    progressSummary
  };
}

export function isRuntimeMainlineRunning(runtime: RuntimeState | null): boolean {
  return getRuntimeMainlineStatus(runtime).running;
}
