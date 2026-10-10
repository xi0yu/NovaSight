import { expect, it } from "vitest";
import { buildControlTrace, type BuildControlTraceInput } from "./controlTrace";

it("presents an empty output consumer without requiring a physical kmNet connection", () => {
  const input: BuildControlTraceInput = {
    runtimeRunning: true, detectionBatchFps: 60, publishedBatches: 1, consumedBatches: 1, targetingBatches: 1,
    detectionDataAgeMs: 1, freshnessThresholdMs: 50, detectionCount: 1, rawCandidateCount: 1,
    eligibleCandidateCount: 1, selectedTargetCount: 1, trackId: 1, classLabel: "0", targetScore: 0.9,
    observedAim: "1,1", controlAim: "1,1", controlCenter: "0,0", errorDistancePx: 1,
    controlFrameAgeMs: 1, measurementDtMs: 16, fireDelayElapsedMs: null, fireDelayRemainingMs: null,
    triggerActive: null, kmnetConnectionLabel: "未连接", deviceLastError: "", nullMouseProtocol: true, outputEnabled: true, willEmit: true, kmnetRuntimeConnected: false,
    hardwareTriggerRequired: false, hasTarget: true, maxOutputXCounts: 100, maxOutputYCounts: 100,
    targetPipelineMessage: "", targetPipelineRejections: "", targetPipelineStage: "", targetPipelineCode: "",
    controllerActive: true, controllerMode: "Atan", movementStrategy: "", controlError: "1,1", integerCommand: "1,1", fullError: "1,1", floatDemand: "1,1", noSendReason: "",
    fireDelayEnabled: false, fireDelayPending: false, fireDelayConfiguredMs: 0, predictionEnabled: false, outputTrace: null,
  };
  const trace = buildControlTrace(input);
  const output = trace.steps.find((step) => step.id === "output")!;
  expect(output.value).toBe("空");
  expect(output.state).toBe("ready");
  expect(output.detail).toContain("静默消费");
});
