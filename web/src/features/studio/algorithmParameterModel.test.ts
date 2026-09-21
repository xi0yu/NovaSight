import { describe, expect, it } from "vitest";

import {
  TARGETING_PIPELINE_FIELDS,
  buildTargetingParameterGroups,
  type TargetingParameterValues
} from "./algorithmParameterModel";

const values: TargetingParameterValues = {
  targetMinConfidence: 0.5,
  candidateRatioMaxAspect: 6,
  targetSwitchPreferenceAdvantage: 0.08,
  targetSwitchContinuityScore: 0.7,
  targetSwitchDelayMs: 50,
  trackerMaxMatchDistance: 1.5,
  trackerPositionCostWeight: 0.75,
  trackerIouCostWeight: 0.25,
  trackerScaleCostWeight: 0.15,
  trackerClassCostWeight: 0.35,
  trackerMaxSizeRatio: 2.5,
  trackerMaxAssociationDtMs: 150,
  targetLostGraceMs: 120,
  trackerKalmanAccelerationNoise: 1200,
  trackerKalmanMeasurementNoiseX: 16,
  trackerKalmanMeasurementNoiseY: 16,
  trackerKalmanMaxPredictDtMs: 35,
  trackerKalmanMaxPredictMissingMs: 80,
  trackerKalmanMaxPredictSteps: 5,
  trackerKalmanNisThreshold: 9.21,
  trackerKalmanNisHardReject: 16,
  targetSelectionMotionHorizonMs: 30
};

describe("target decision parameter model", () => {
  it("exposes the replacement policy without the retired class ratio", () => {
    expect(TARGETING_PIPELINE_FIELDS).toContain("target_selection_motion_weight");
    expect(TARGETING_PIPELINE_FIELDS).toContain("tracker_class_cost_weight");
    expect(TARGETING_PIPELINE_FIELDS).not.toContain("target_selection_class_ratio" as never);

    const groups = buildTargetingParameterGroups(values);
    expect(groups.targetAdvancedParameters.map(({ key }) => key)).toContain(
      "target_selection_motion_horizon_ms"
    );
    expect(groups.trackerCoreParameters.map(({ key }) => key)).toContain(
      "tracker_class_cost_weight"
    );
  });
});
