import type { RuntimeState } from "../../api";

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
  controlObservations: number | null;
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

function readOptionalNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function firstNumber(...values: unknown[]): number | null {
  for (const value of values) {
    const number = readOptionalNumber(value);
    if (number !== null) {
      return number;
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
  const pipeline = asRecord(runtime?.pipeline);
  const deepstream = asRecord(pipeline.deepstream);
  const inference = asRecord(runtime?.inference);
  const statistics = asRecord(runtime?.statistics);
  const fatal = asRecord(runtime?.fatal_error);
  const terminalError = readBoolean(inference.terminal_error) || readBoolean(deepstream.terminal_error);
  const pipelineLastError =
    readString(pipeline.last_error) || (terminalError ? readString(deepstream.last_error) : "");
  const fatalMessage = readString(fatal.message);
  const failureMessage =
    pipelineLastError ||
    fatalMessage ||
    (terminalError ? readString(inference.detail) || readString(inference.reason) : "");
  const nvinferInputFrames = firstNumber(
    statistics.nvinfer_input_counter,
    deepstream.input_frames,
    inference.input_frames
  );
  const metadataExtractions = firstNumber(
    deepstream.metadata_extractions,
    inference.metadata_extractions
  );
  const publishedBatches = firstNumber(
    statistics.detection_batch_counter,
    deepstream.published_batches,
    inference.published_batches
  );
  const consumedBatches = firstNumber(
    statistics.detection_batch_consumed_counter,
    pipeline.consumed_detection_batches
  );
  const controlObservations = firstNumber(
    statistics.control_observation_counter,
    pipeline.control_observations
  );
  const hasInferenceSignal =
    positive(nvinferInputFrames) ||
    positive(metadataExtractions) ||
    positive(publishedBatches);
  const hasRuntimeConsumption =
    positive(consumedBatches) || positive(controlObservations);
  const resolvedFailureMessage = failureMessage;
  const progressSummary = [
    `nvinfer=${counterLabel(nvinferInputFrames)}`,
    `metadata=${counterLabel(metadataExtractions)}`,
    `published=${counterLabel(publishedBatches)}`,
    `consumed=${counterLabel(consumedBatches)}`,
    `control=${counterLabel(controlObservations)}`
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
    nvinferInputFrames,
    metadataExtractions,
    publishedBatches,
    consumedBatches,
    controlObservations,
    hasInferenceSignal,
    hasRuntimeConsumption,
    progressSummary
  };
}

export function isRuntimeMainlineRunning(runtime: RuntimeState | null): boolean {
  return getRuntimeMainlineStatus(runtime).running;
}
